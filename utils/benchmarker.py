from image_metrics import process


# test_filenames = ["./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_1spp_baseline.png", 
#                   "./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_1spp_ours.png", 
#                   "./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_64spp_baseline.png", 
#                   "./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_64spp_ours.png", 
#                   "./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_256spp_baseline.png", 
#                   "./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_256spp_ours.png"] 

test_filenames = ["./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_16spp_baseline.png", 
                  "./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_16spp_ours.png"]

gt_filename = ["./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_gt.png"] * len(test_filenames)

process(zip(gt_filename, test_filenames))