from Preprocessing.LoadData import (sample_from_main)

part_dict = sample_from_main(part=5)  # part_dict[0], part_dict[1], ... would be arrays after standardization
print(part_dict)

for i in part_dict.keys():
    pass



