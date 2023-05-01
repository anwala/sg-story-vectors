# Story Vectors
Experimental code to replace [storygraphbot](https://github.com/oduwsdl/storygraphbot) with new vector-based clusterer 

# Installation
Requirements:
* [Networkx](https://networkx.org/documentation/stable/install.html)
* [BLOC](https://github.com/anwala/bloc) (From Cluster branch):
```
$ git clone -b cluster https://github.com/anwala/bloc.git
$ pip install bloc
```
* [storygraph-toolkit](https://github.com/oduwsdl/storygraph-toolkit)

# Example
```
from sg_story_vects import cluster_stories_for_dates
story_vectors = cluster_stories_for_dates(['2021-01-06', '2021-01-07'], min_cosine_sim=0.65, story_vect_dim=1000, cmp_only_event_con_comps=False)
```