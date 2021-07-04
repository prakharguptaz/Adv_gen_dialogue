# Classification commands

Sample train command (which also runs accuracy test)
```
python run_dialclassifier.py --given_seed 102 --model_name_or_path MLM_PRETRAINED_MODEL_PATH --output_dir tmp/OUTPUT_FOLDER/ --train_file ../dataset/TRAIN_FILE --validation_file ../dataset/DEV_FILE --test_file ../dataset/TEST_FILE --per_device_train_batch_size 60 --per_device_eval_batch_size 60  --type_negative_test adversarial_negative_responses --overwrite_cache --type_negative both --type_negative_extra  JSON_KEY_ADVERSARIAL_NEGATIVE --do_eval --do_predict --do_train --load_best_model_at_end --metric_for_best_model accuracy  --overwrite_output_dir --overwrite_cache --evaluation_strategy steps --eval_steps 200 --save_steps 400 --num_train_epochs 3 --type_negative_extra_max 5 --gradient_accumulation_steps 1
```

Sample command for classification testing
```
python run_dialtest.py --model_name_or_path tmp/OUTPUT_FOLDER/ --output_dir tmp/OUTPUT_FOLDER/  --validation_file ../dataset/DEV_FILE --test_file ../dataset/TEST_FILE --per_device_train_batch_size 60 --per_device_eval_batch_size 60  --type_negative_test adversarial_negative_responses --overwrite_cache  --do_predict    --overwrite_output_dir --overwrite_cache  
```

Sample command for ranking testing
```
python run_ranktest.py --model_name_or_path tmp/OUTPUT_FOLDER/ --output_dir tmp/OUTPUT_FOLDER/  --validation_file ../dataset/DEV_FILE --test_file ../dataset/TEST_FILE --per_device_train_batch_size 60 --per_device_eval_batch_size 60  --type_negative_test adversarial_negative_responses --overwrite_cache  --do_predict    --overwrite_output_dir --overwrite_cache  
```
