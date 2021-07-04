# keyword based approach

Requirements:
tranformers==3.5.0
pytorch-ignite==0.3.0

To train use command 
``` python train_keyword.py --train_file $train_file_path --validation_file $validation_file_path ```

To generate
``` python generate_sim.py --model_checkpoint $model_path --input_file $input_file_path ```

Sample files are present in the data folder. The input file for training is a csv file where each row consists of one sentence of a dialogue turn, that is, if a dialogue turn has multiple sentences, the csv will contain each sentence in one row which is aggregated using pandas library. The format of this input data can be changed and data reader code can be adjusted.

To use keyword-context model, remove keyphrases_neighbours from the code

