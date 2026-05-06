Setup for processing:
Clone the STAMP repository at https://github.com/KatherLab/STAMP/ and follow the installation instructions.

Data:

3 datasets from Auckland, Waikato, and Canterbury and a spreadsheet with the patient numbers, slide numbers and labels.


Preprocessing:

I used the STAMP preprocessing pipeline found at at https://github.com/KatherLab/STAMP/. 

The setup was done following the STAMP readme and getting_started instructions. 
The preprocessing was done according to the instructions also - 
Copy the .yaml template from STAMP and populate it with the right paths and fields and then run the stamp  --preprocess command


NOTE: To use the CONCH model (or likely any other model) for feature extraction it is likely you will require to request access to the model on Huggingface and authenticate before running stamp --preprocess. 
CONCH: https://huggingface.co/MahmoodLab/CONCH


Data Splits:

I copied my extracted feature vectors to individual folders to only have to do it once and I can't find the code I used to do it but I used the spreadsheet of patient numbers, slide numbers and labels with sklearn StratifiedGroupKFold to group by patient number and label by label and do the split, I did k=5 and used the first split to separate ~20% into the test set and then the remaining ~80% got split into 5 folds. The train/test split wasn't perfectly 80/20 due to the group stratification, and the 5 folds of the train set weren't perfectly even splits either for the same reason. 


Training: 

I used train.py from the command line.
e.g.

python train.py --model_name "automil" --num_epochs 30

After training I wrote a .txt file with the ckpt names for each fold. called ckpts.txt to be used for plotting because I wanted to avoid the os library as much as possible, Posix paths were really annoying.

Validation and Test plotting: 

After that I ran plot() in val_test_plots.py to get the validation plots and metrics.
Then after looking at which folds were the best I wrote test_models.txt with the paths of the best folds repeated 5 times to avoid redoing the logic for getting the paths. (Waste of compute, takes 5 times longer to run than necessary but saved me from having to write good code.)

Then ran plot(test=True) in val_test_plots.py for the test set results, the only important plot is plots/test/best.png, the others are demonstrations of wasted compute.


Heatmaps:

I looped through dataset.featfiles to find which indices that correspond to which slides. Then found the path(s) of the corresponding WSIs (The actual .svs images, not the feature vectors).

Then ran heatmaps.py with the (index, path) pairs for a each slide. 




Aside: 
I am quite confident this is all of the necessary files to get things going but it may not be, my file tracking is an absolute mess (there was a lot of trial and a lot of error to get here).
If something is missing let me know at oscar.lindsay@gmail.com and I can try to find it.