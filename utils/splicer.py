from image_splice import splice_images

test_filenames = ["./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_16spp_baseline.png", 
                  "./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_16spp_ours.png"]

gt_filename = ["./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_gt.png"]

splice_images(test_filenames + gt_filename, border_angle=10, output_filename='spliced.png')