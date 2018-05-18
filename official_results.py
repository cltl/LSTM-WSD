import result_tables
import os


input_folder = 'output'

out_dir = 'paper_tables'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

experiments = [('synset---se2---semcor', 'synset---se13---semcor'),
               ('synset---se2---omsti', 'synset---se13---omsti'),
               ('synset---se2---semcor_omsti', 'synset---se13---semcor_omsti'),
               #('synset-se2-framework-semcor_mun-lp', 'synset-se13-framework-semcor_mun-lp')
               ]
result_tables.p_r_f1_mfs_lfs(input_folder, experiments, 'paper_tables/mfs_lfs.tex')

result_tables.f1(input_folder, experiments, 'paper_tables/overall.tex')

result_tables.strategies(input_folder, experiments, 'paper_tables/strategies.tex')

result_tables.coverage(input_folder, experiments, 'paper_tables/coverage.tex')

experiments = [('sensekey---se2---semcor', 'sensekey---se13---semcor'),
               ('sensekey---se13---omsti', 'sensekey---se13---omsti'),
               ('sensekey---se2---semcor_omsti', 'sensekey---se13---semcor_omsti')
               ]

result_tables.sensekey(input_folder, experiments, 'paper_tables/sensekey.tex')

result_tables.p_r_f1_mfs_lfs(input_folder, experiments, 'paper_tables/sensekey_mfs_lfs.tex')
