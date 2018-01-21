# BASISS
Broad Area Satellite Imagery Semantic Segmentation

==============================================================================

Code Overview

This package segments satellite imagery over large swaths of land (or sea).  The included examples use [SpaceNet](https://spacenetchallenge.github.io/) labels to identify roads in high resolution satellite imagery.  Access to a GPU is required for training, though inference will function (slowly) on a CPU.


==============================================================================

1. Installation

		A. Install nvidia-docker
		B. Build container
			nvidia-docker build -t basiss path_to_basiss/docker
		C. Download scripts
			download this github repo (or put it in the docker file...)
		D. Run container
			nvidia-docker run -it -v /raid:/raid --name bassiss_train basiss		


==============================================================================

2.	Download SpaceNet data, see the [instructions](https://github.com/SpaceNetChallenge/utilities/tree/master/content/download_instructions).


==============================================================================

3.	Create training masks.  The script used here is a lightly modified version of the code described in [Blog1](https://medium.com/the-downlinq/creating-training-datasets-for-the-spacenet-road-detection-and-routing-challenge-6f970d413e2f).  Execute these scripts in a unique conda environment; conda install insctuctions are [here](https://conda.io/miniconda.html). The commands below create the training images; replace "train" with "test" to create testing images.

	basiss_path=/raid/local/src/basiss
	cd $basiss_path/src
	conda env create -f apls_environment.yml   # to deactivate environment: source deactivate
	source activate apls_environment

	#for details on arguments type: python create_spacenet_masks.py --help
	python $basiss_path/create_spacenet_masks.py \
	    --path_data=/path_to_spacenet_data/AOI_2_Vegas_Train \
	    --output_df_path=$basiss_path/packaged_data/AOI_2_Train_2m_file_locs.csv \
	    --buffer_meters=2 \
	    --n_bands=3 \
	    --make_plots=0 \
	    --overwrite_ims=0

![Alt text](/example_ims/mask_img998.png?raw=true "Figure 1")


==============================================================================

4.	Train a model

		# train Las Vegas SpaceNet 3-band data with unet, and sliced into 400 
		#  	pixel cutouts
		basiss_path=/raid/local/src/basiss
		outname=AOI_2_Vegas_unet_2m_train
		cd $basiss_path
		nohup python -u src/basiss.py \
			--path $basiss_path \
			--model unet \
			--mode train \
			--file_list AOI_2_Train_2m_file_locs.csv \
			--slice_x 400 --slice_y 400 \
			--stride_x 300 --stride_y 300 \
			--n_bands 3 \
			--n_classes 2 \
			--batchsize 32 \
			--validation_split 0.1 \
			--early_stopping_patience 4 \
			--epochs 128 \
			--gpu 0 \
			--prefix $outname > \
				results/$outname.log & tail -f results/$outname.log


==============================================================================

5. Test on images of arbitrary size

		# test Las Vegas SpaceNet 3-band data with unet, and sliced into 400 
		#  	pixel cutouts
		basiss_path=/raid/local/src/basiss
		outname=AOI_2_Vegas_unet_2m_test
		cd $basiss_path
		nohup python -u src/basiss.py \
			--path $basiss_path \
			--model unet \
			--mode test \
			--file_list massive_file_list.csv \
			--model_weights AOI_2_Vegas_unet_2m_train_model_best.hdf5 \
			--slice_x 400 --slice_y 400 \
			--stride_x 300 --stride_y 300 \
			--n_bands 3 \
			--n_classes 2 \
			--batchsize 16 \
			--gpu 3 \
			--prefix $outname > \
				results/$outname.log & tail -f results/$outname.log

![Alt text](/example_ims/unet0.png?raw=true "Figure 2")

