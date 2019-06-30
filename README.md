# halo-formation
Several scripts for investigating the relation between galaxy observables and halo properties (in LGalaxies data)

"plotting_observables" contains code to plot observable parameters (e.g. sm, srf) vs halo properties (hm, formtion time), with some simple tests

"running_random_forests" runs random forests on several paramters to predict formation times and plots the output. This is compared to formation times predicted using just 1 parameter

"data" would contain the catalogues used in the study. SQL queries used to get this data for LGalaxies using casjobs are stored in "queries.txt"
