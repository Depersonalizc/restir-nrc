from image_metrics import process
from image_splice import splice_images


test_filenames = ["./data/mdl_simple_watercolor_benchmark/mdl_demo_1spp_ours.png", 
                  "./data/mdl_simple_watercolor_benchmark/mdl_demo_16spp_ours.png",  
                  "./data/mdl_simple_watercolor_benchmark/mdl_demo_64spp_ours.png", 
                  "./data/mdl_simple_watercolor_benchmark/mdl_demo_256spp_ours.png",
                  "./data/mdl_simple_watercolor_benchmark/mdl_demo_1spp_baseline.png", 
                  "./data/mdl_simple_watercolor_benchmark/mdl_demo_16spp_baseline.png",  
                  "./data/mdl_simple_watercolor_benchmark/mdl_demo_64spp_baseline.png", 
                  "./data/mdl_simple_watercolor_benchmark/mdl_demo_256spp_baseline.png"]

def splice_filenames(spp):
    # return [f"./data/mdl_simple_bistro_benchmark/benchmark_bistro_{spp}spp_baseline.png",
    #         f"./data/mdl_simple_bistro_benchmark/benchmark_bistro_{spp}spp_ours.png",
    #         f"./data/mdl_simple_bistro_benchmark/benchmark_bistro_gt.png",]
    
    return [f"./data/mdl_simple_watercolor_benchmark/mdl_demo_{spp}spp_baseline.png",
            f"./data/mdl_simple_watercolor_benchmark/mdl_demo_{spp}spp_ours.png",
            f"./data/mdl_simple_watercolor_benchmark/mdl_demo_gt.png",]
     
    # return [f"./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_{spp}spp_baseline.png",
    #         f"./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_{spp}spp_ours.png",
    #         f"./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_gt.png",] 

# test_filenames = ["./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_16spp_baseline.png", 
#                   "./data/mdl_simple_cornell_benchmark/mdl_simple_cornell_16spp_ours.png"]

gt_filename = ["./data/mdl_simple_watercolor_benchmark/mdl_demo_gt.png"] * len(test_filenames)

# process(zip(gt_filename, test_filenames))

# splice_images(splice_filenames(1), border_angle=8, output_filename='spliced-watercolor-1spp.png')
splice_images(splice_filenames(16), border_angle=13, output_filename='spliced-watercolor-16spp.png')