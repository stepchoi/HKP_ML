import gc

from Preprocessing.LoadData import (load_data, clean_set)

'''
part_dict = sample_from_main(part=5)  # part_dict[0], part_dict[1], ... would be arrays after standardization
print(part_dict)

for i in part_dict.keys():
    pass
'''

main = load_data(sql_version=False)  # change sql_version -> True if trying to run this code through Postgres Database
period_1 = dt.datetime(2008, 3, 31)
main_period = clean_set(main, period_1)
train_x, test_x = main_period.standardize_x(return_test_x = True)

del main_period  # delete this train_x and collect garbage -> release memory
gc.collect()



