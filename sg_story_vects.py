import sys
import networkx as nx

from bloc.util import conv_tf_matrix_to_json_compliant
from bloc.util import cosine_sim
from bloc.util import get_tf_matrix
from itertools import combinations
#from bloc.util import dumpJsonToFile
#from bloc.util import getDictFromJsonGZ

from storygraph_tk.maxgraphs import get_timezone_aware_graphs
from storygraph_tk.util import parallelTask

def get_con_comps_vocab_dist(cc, graphs):
    
    if( 'graph-index' not in cc or len(graphs) == 0 ):
        return {}

    doc_lst = []
    cc_ents = [ graphs[cc['graph-index']]['nodes'][n]['entities'] for n in cc['nodes'] ]
    
    for cce in cc_ents:
        node_doc = ' '.join([e['entity'] for e in cce])
        doc_lst.append({ 'text': node_doc, 'id': len(doc_lst) })
    
    count_vectorizer_kwargs = {'binary': True}
    tf_mat = get_tf_matrix(doc_lst, n=1, min_df=2, set_top_ngrams=True, top_ngrams_add_all_docs=True, count_vectorizer_kwargs=count_vectorizer_kwargs)
    
    return {t['term']: t['doc_rate'] for t in tf_mat['top_ngrams']['all_docs']}

def cmp_graph_con_comps(fst_vocab_dist, sec_vocab_dist):
    
    fst_cc_vect = []
    sec_cc_vect = []

    common_vocab = set(list(fst_vocab_dist.keys()) + list(sec_vocab_dist.keys()))
    for term in common_vocab:
        fst_cc_vect.append( fst_vocab_dist.get(term, 0) )
        sec_cc_vect.append( sec_vocab_dist.get(term, 0) )
    
    return cosine_sim( [fst_cc_vect], [sec_cc_vect] )

def gen_story_vect_graph(all_con_comps, pairs, min_cc_avg_degree=0, cmp_sibling_con_comps=True, cmp_only_event_con_comps=True, min_cosine_sim=0.75):
    
    story_vect_g = nx.Graph()
    [ story_vect_g.add_node(i) for i in range(len(all_con_comps)) ]
    
    for fst_cc_indx, sec_cc_indx in pairs:
        
        fst_cc = all_con_comps[fst_cc_indx]
        sec_cc = all_con_comps[sec_cc_indx]
        
        if( fst_cc['input']['args']['cc']['graph-index'] == sec_cc['input']['args']['cc']['graph-index'] and cmp_sibling_con_comps is False ):
            continue

        if( fst_cc['input']['args']['cc']['avg-degree'] < min_cc_avg_degree or sec_cc['input']['args']['cc']['avg-degree'] < min_cc_avg_degree ):
            continue

        cosine_sim = cmp_graph_con_comps(fst_cc['output'], sec_cc['output'])
        
        if( cosine_sim >= min_cosine_sim ):
            story_vect_g.add_edge(fst_cc_indx, sec_cc_indx)

    return story_vect_g

def get_max_node_indx(cc_nodes, links):
    
    cc = nx.Graph()
    for l in links:
        if( l['source'] in cc_nodes and l['target'] in cc_nodes ):
            cc.add_edge(l['source'], l['target'])

    return max(cc.nodes, key=cc.degree)

def get_story_vect_struct():
    return {'titles': {}, 'con_comps': [], 'dates': [], 'total_docs': 0, 'topic_dist': []}

def get_vects_frm_con_comps(graphs, cmp_only_event_con_comps=True):
    
    all_con_comps = []
    graph_count = len(graphs)

    for i in range(graph_count):
        for j in range( len(graphs[i]['connected-comps']) ):
            
            cc = graphs[i]['connected-comps'][j]
            if( cmp_only_event_con_comps is True and cc['node-details']['connected-comp-type'] != 'event' ):
                continue
            
            cc['graph-index'] = i
            cc['cc-index'] = j
            all_con_comps.append({
                'func': get_con_comps_vocab_dist,
                'args': {'cc': cc, 'graphs': graphs},
                'misc': None,
                'print': '\tgen cc vect ({}), graph {} of {}'.format(len(all_con_comps), i+1, graph_count) if len(all_con_comps) % 50 == 0 else ''
            })

    return parallelTask(all_con_comps)

def cmp_story_vects(fst_story_vect, sec_story_vect):
    
    if( 'topic_dist' not in fst_story_vect or 'topic_dist' not in sec_story_vect ):
        return -1
    
    fst_t_df = { t[0]: t[1] for t in fst_story_vect['topic_dist'] }
    sec_t_df = { t[0]: t[1] for t in sec_story_vect['topic_dist'] }
    
    fst_vect = []
    sec_vect = []
    common_vocab = set(list(fst_t_df.keys()) + list(sec_t_df.keys()))
    for term in common_vocab:
        fst_vect.append( fst_t_df.get(term, 0)/fst_story_vect['total_docs'] )
        sec_vect.append( sec_t_df.get(term, 0)/fst_story_vect['total_docs'] )
    
    return cosine_sim( [fst_vect], [sec_vect] )

def cluster_stories_frm_graphs(graphs, date, min_cc_avg_degree=0, cmp_sibling_con_comps=True, cmp_only_event_con_comps=True, min_cosine_sim=0.75, story_vect_dim=1000, **kwargs):

    '''
        Input:
            Graphs (e.g., 144) representing snapshots of the news cycle across a single day. Each graph features nodes (news articles), edges connecting news articles, and connected components (representation of news story.)
        Process:
            All connected components are extracted across all graphs and converted to story vectors with get_vects_frm_con_comps()
            Create an undirected graph G (node: cc story vector, edges: similary nodes) that connects all pairs of similar story vectors with gen_story_vect_graph()
            All connected components in G represent the cluster of connected components (represented by their respective story vectors) that are part of the same story
            For a single story (e.g., story_vect_conn_comp), create a new vector (e.g., story_v['story_vectors'][0]) over all the entities in the connected components of the story.
        Output:
            A list of story vectors
    '''
    graph_uri_prefix = kwargs.get('graph_uri_prefix', '')
    print('\ncluster_stories_frm_graphs():', date)

    story_v = {'story_vectors': []}
    
    #all_con_comps: [{'input': , 'output': , 'misc'}, ...], where input is content of all_con_comps
    all_con_comps = get_vects_frm_con_comps(graphs, cmp_only_event_con_comps)
    cc_count = len(all_con_comps)
    indices = list(range(cc_count))     
    pairs = combinations(indices, 2)
    print('\ttotal ccs:', cc_count)
    
    print('\tgen_story_vect_graph - start')
    story_vect_g = gen_story_vect_graph(all_con_comps, pairs, min_cc_avg_degree=min_cc_avg_degree, cmp_sibling_con_comps=cmp_sibling_con_comps, cmp_only_event_con_comps=cmp_only_event_con_comps, min_cosine_sim=min_cosine_sim)
    story_vect_g = nx.connected_components(story_vect_g)
    print('\tgen_story_vect_graph - done\n')

    print('\tgen_story_vect - start')
    for story_vect_conn_comp in story_vect_g:
        
        #story_vect_conn_comp: set of connected components that map to the same story
        #print('\nstory:', story_vect_conn_comp)

        doc_lst = []
        story_vect = get_story_vect_struct()
        story_vect['dates'].append(date)
        
        for storygraph_cc in story_vect_conn_comp:

            con_comp = all_con_comps[storygraph_cc]['input']['args']['cc']
            graph_indx = con_comp['graph-index']
            cc_nodes = con_comp['nodes']

            max_deg_node = get_max_node_indx(cc_nodes, graphs[graph_indx]['links'])
            max_deg_node_title = graphs[graph_indx]['nodes'][max_deg_node]['title']
            graph_uri = '{}t={}&cursor={}&hist=144'.format( graph_uri_prefix, graphs[graph_indx]['timestamp'].split('.')[0], graphs[graph_indx]['cursor'] )

            #print(max_deg_node_title)
            #print(con_comp)
            #print(graph_uri)
            #print()

            story_vect['titles'].setdefault(max_deg_node_title, 0)
            story_vect['titles'][max_deg_node_title] += 1
            story_vect['con_comps'].append({ 'graph_uri': graph_uri, 'index': con_comp['cc-index'], 'avg_degree': con_comp['avg-degree'] })

            cc_ents = [ graphs[graph_indx]['nodes'][n]['entities'] for n in cc_nodes ]
            for cce in cc_ents:
                node_doc = ' '.join([e['entity'] for e in cce])
                doc_lst.append({ 'text': node_doc, 'id': len(doc_lst) })
            
            story_vect['total_docs'] = len(doc_lst)

        tf_mat = get_tf_matrix(doc_lst, n=1, min_df=2, set_top_ngrams=True, top_ngrams_add_all_docs=True, count_vectorizer_kwargs={'binary': True})
        
        
        story_vect['con_comps'] = sorted(story_vect['con_comps'], key=lambda x: x['graph_uri'])
        story_vect['topic_dist'] = tf_mat['top_ngrams']['all_docs'][:story_vect_dim]
        story_vect['topic_dist'] = [ [t['term'], t['doc_freq']] for t in story_vect['topic_dist'] ]
        story_v['story_vectors'].append(story_vect)

    print('\tgen_story_vect - done\n')

    return story_v

def obsolete_single_day_story_clusterer(date, min_cc_avg_degree=0, cmp_sibling_con_comps=True, cmp_only_event_con_comps=True, min_cosine_sim=0.75, story_vect_dim=1000):

    graphs = get_timezone_aware_graphs( date, f'{date} 00:00:00', f'{date} 23:59:59', graph_url='https://archive.org/download/', graph_location='usa', timezone='US/Eastern' )
    ##dumpJsonToFile('sample_graphs.json', graphs, indentFlag=False)
    #graphs = getDictFromJsonGZ('sample_graphs.json.gz')

    graphs = [g['graph'] for g in graphs['graphs']]
    graph_uri_prefix = 'https://web.archive.org/storygraph/graphs/usa/#'
    day_story_vect = cluster_stories_frm_graphs(graphs, date=date, min_cc_avg_degree=min_cc_avg_degree, cmp_sibling_con_comps=cmp_sibling_con_comps, cmp_only_event_con_comps=cmp_only_event_con_comps, min_cosine_sim=min_cosine_sim, story_vect_dim=story_vect_dim, graph_uri_prefix=graph_uri_prefix)

    return day_story_vect

#dates: list of dates (e.g., ["YYYY-MM-DD"])
def cluster_stories_for_dates(dates, min_cc_avg_degree=0, cmp_sibling_con_comps=True, cmp_only_event_con_comps=True, min_cosine_sim=0.7, story_vect_dim=1000, **kwargs):
    
    '''
        Description:
            Some stories span multiple days. Since clustering happens for individual days separately (e.g., 2023-01-06 and 2023-01-07). The same story could appear in the list of story vectors for each day (e.g., 2023-01-06 and 2023-01-07).
            The purpose of this function is to merge such stories.
        Logic:
            Clustering happens by building an undirected graph G. Nodes represent individual story vectors, and edges connect similar story vectors. Sibling story vectors generated for the same day are not compared since they've already had the chance to be clustered and did not merge.
            Connected components represent stories that straddle multiple days. Singletons are stories that do not.
    '''
    def combine_stories_vects(stories_vects, story_vect_dim):

        story_vect = get_story_vect_struct()
        story_vect['topic_dist'] = {}
        for s in stories_vects:
            
            for title, freq in s['titles'].items():
                story_vect['titles'].setdefault(title, 0)
                story_vect['titles'][title] += freq

            for term, doc_freq in s['topic_dist']:
                story_vect['topic_dist'].setdefault(term, 0)
                story_vect['topic_dist'][term] += doc_freq

            story_vect['dates'] += s['dates']
            story_vect['con_comps'] += s['con_comps']
            story_vect['total_docs'] += s['total_docs']

        story_vect['dates'] = sorted(list(set(story_vect['dates'])))
        story_vect['con_comps'] = sorted(story_vect['con_comps'], key=lambda x: x['graph_uri'])
        story_vect['topic_dist'] = sorted( story_vect['topic_dist'].items(), key=lambda x: x[1], reverse=True )[:story_vect_dim]
        
        return story_vect

    kwargs.setdefault('timezone', 'US/Eastern')
    kwargs.setdefault('graph_location', 'usa')
    kwargs.setdefault('graph_url', 'https://archive.org/download/')
    kwargs.setdefault('graph_uri_prefix', 'https://web.archive.org/storygraph/graphs/usa/#')
    
    timezone = kwargs['timezone']
    graph_location = kwargs['graph_location']
    graph_url = kwargs['graph_url']
    graph_uri_prefix = kwargs['graph_uri_prefix']

    multi_day_story_vects = []
    for d in dates:
        #day_story_vect: contains story vectors (topic distribution) for a single day
        #day_story_vect = single_day_story_clusterer(date=d, min_cc_avg_degree=min_cc_avg_degree, cmp_sibling_con_comps=cmp_sibling_con_comps, cmp_only_event_con_comps=cmp_only_event_con_comps, min_cosine_sim=min_cosine_sim, story_vect_dim=story_vect_dim)
        graphs = get_timezone_aware_graphs( d, f'{d} 00:00:00', f'{d} 23:59:59', graph_url=graph_url, graph_location=graph_location, timezone=timezone )
    
        graphs = [g['graph'] for g in graphs['graphs']]
        day_story_vect = cluster_stories_frm_graphs(graphs, date=d, min_cc_avg_degree=min_cc_avg_degree, cmp_sibling_con_comps=cmp_sibling_con_comps, cmp_only_event_con_comps=cmp_only_event_con_comps, min_cosine_sim=min_cosine_sim, story_vect_dim=story_vect_dim, graph_uri_prefix=graph_uri_prefix)
        multi_day_story_vects += day_story_vect['story_vectors']

    
    sv_count = len(multi_day_story_vects)
    indices = list(range(sv_count))     
    pairs = combinations(indices, 2)

    multi_day_stories = nx.Graph()
    [ multi_day_stories.add_node(i) for i in range(sv_count) ]

    for fst_sv_indx, sec_sv_indx in pairs:
        
        fst_story_vect = multi_day_story_vects[fst_sv_indx]
        sec_story_vect = multi_day_story_vects[sec_sv_indx]

        #don't compare story vectors from the same day since they've already had the chance to be clustered
        if( fst_story_vect['dates'][0] == sec_story_vect['dates'][0] ):
            continue

        cosine_sim = cmp_story_vects(fst_story_vect, sec_story_vect)
        if( cosine_sim < min_cosine_sim ):
            continue

        multi_day_stories.add_edge(fst_sv_indx, sec_sv_indx)
    

    final_stories_vectors = []
    multi_day_stories = nx.connected_components(multi_day_stories)
    for multi_day_stories_cc in multi_day_stories:
        
        #multi_day_stories_cc: collection of story_vectors for single or multiple days
        multi_day_stories_cc = list(multi_day_stories_cc)

        if( len(multi_day_stories_cc) == 1 ):
            #This story does not straddle multiple days
            final_stories_vectors.append( multi_day_story_vects[multi_day_stories_cc[0]] )
            continue

        new_story_vect = combine_stories_vects([multi_day_story_vects[story_v_indx] for story_v_indx in multi_day_stories_cc], story_vect_dim)
        final_stories_vectors.append(new_story_vect)
        
    
    final_stories_vectors = {
        'story_vectors': final_stories_vectors,
        'self': {
            'dates': dates,
            'min_cc_avg_degree': min_cc_avg_degree,
            'cmp_sibling_con_comps': cmp_sibling_con_comps,
            'cmp_only_event_con_comps': cmp_only_event_con_comps,
            'min_cosine_sim': min_cosine_sim,
            'story_vect_dim': story_vect_dim,
            **kwargs
        }   
    }
    
    #dumpJsonToFile('final_stories_vectors.json', final_stories_vectors)
    return final_stories_vectors


'''
cluster_stories_for_dates(['2021-01-06', '2021-01-07'], min_cosine_sim=0.65, story_vect_dim=1000, cmp_only_event_con_comps=False)
'''