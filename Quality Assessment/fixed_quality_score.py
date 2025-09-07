import argparse
import os
import sys

from evaluation.QualityModel import QualityModel


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default='/data/quality_data',
                        help='Root dir for evaluation dataset')
    parser.add_argument('--pairs', type=str, default='pairs.txt',
                        help='lfw pairs.')
    parser.add_argument('--datasets', type=str, default='custom_dataset',
                        help='list of evaluation datasets (,)  e.g.  custom_dataset, XQLFW, lfw,calfw,agedb_30,cfp_fp,cplfw,IJBC.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id.')
    # Fixed model path to match your structure
    parser.add_argument('--model_path', type=str, default="./CR-FIQAL",
                        help='path to pretrained evaluation.')
    parser.add_argument('--model_id', type=str, default="181952",
                        help='digit number in backbone file name')
    parser.add_argument('--backbone', type=str, default="auto",
                        help=' iresnet100, iresnet50, or auto to detect from weights ')
    parser.add_argument('--score_file_name', type=str, default="custom_quality_scores.txt",
                        help='score file name, the file will be store in the same data dir')
    parser.add_argument('--color_channel', type=str, default="RGB",  # Changed to RGB
                        help='input image color channel, two option RGB or BGR')

    return parser.parse_args(argv)


def read_image_list(image_list_file, image_dir=''):
    image_lists = []
    absolute_list = []
    
    print(f"Reading image list from: {image_list_file}")
    
    with open(image_list_file) as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            absolute_list.append(line)
            # Create full path to image
            full_path = os.path.join(image_dir, line)
            image_lists.append(full_path)
    
    print(f"Found {len(image_lists)} images to process")
    return image_lists, absolute_list

d_data_dir = "data"
def main(param):
    datasets = param.datasets.split(',')
    
    print(f"Initializing quality model from: {param.model_path}")
    # QualityModel expects: model_prefix, model_epoch, gpu_id
    face_model = QualityModel(param.model_path, param.model_id, param.gpu_id)
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        
        # Paths for your data
        dataset_dir = os.path.join(param.data_dir, dataset)
        image_list_file = os.path.join(dataset_dir, 'image_path_list.txt')
        
        if not os.path.exists(image_list_file):
            print(f"Error: {image_list_file} not found!")
            continue
            
        # Read image list
        image_list, absolute_list = read_image_list(image_list_file, d_data_dir)
        
        print(f"Getting quality scores for {len(image_list)} images...")
        
        # Get embeddings and quality scores
        embedding, quality = face_model.get_batch_feature(
            image_list, 
            batch_size=16, 
            color=param.color_channel
        )
        
        # Ensure output directory exists
        output_dir = os.path.join(param.data_dir, dataset)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Write quality scores
        quality_file_path = os.path.join(output_dir, param.score_file_name)
        print(f"Writing quality scores to: {quality_file_path}")
        
        with open(quality_file_path, "w") as quality_score:  # Changed from "a" to "w"
            for i in range(len(quality)):
                quality_score.write(f"{absolute_list[i]} {quality[i][0]}\n")
        
        print(f"Quality assessment complete! Results saved to {quality_file_path}")


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))