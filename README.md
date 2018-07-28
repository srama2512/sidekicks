# Sidekick Policy Learning
This repository contains code and data for the paper 

[Sidekick Policy Learning for Active Visual Exploration](http://vision.cs.utexas.edu/projects/sidekicks/)  
Santhosh K. Ramakrishnan, Kristen Grauman  
ECCV 2018


## Setup
- First install anaconda and setup a new environment. Install anaconda from: https://www.anaconda.com/download/

```
conda create -n spl python=2.7
source activate spl
```
- Clone this project repository and setup requirements using pip.

```
git clone https://github.com/srama2512/sidekicks.git
cd sidekicks
pip install -r requirements.txt
```

- Download preprocessed SUN360 and ModelNet data.

```
wget http://vision.cs.utexas.edu/projects/sidekicks/data.zip
unzip data.zip
```

- Sidekick scores for `ours-rew`, `ours-demo`, `rnd-rewards` on both datasets have been provided [here](http://vision.cs.utexas.edu/projects/sidekicks/scores.zip). The `one-view` model used to generate them have also been provided. 
 
## Evaluating pre-trained models
All the pre-trained models have been provided [here](http://vision.cs.utexas.edu/projects/sidekicks/models.zip). To evaluate them, download them to the `models` directory. To reproduce results from the paper: 

```
wget http://vision.cs.utexas.edu/projects/sidekicks/models.zip
unzip models.zip
sh evaluation_script_final.sh
```

### Evaluation examples
- Evaluating SUN360 `one-view` baseline on the test data with `avg` metric:

```
python eval.py --h5_path data/sun360/sun360_processed.h5 --dataset 0 \
				  --model_path models/sun360/one-view.net --T 1 --M 8 --N 4 \
				  --start_view 2 --save_path dummy/ 
```

- Evaluating SUN360 `ltla` baseline on the test data with `avg` metric:

```
python eval.py --h5_path data/sun360/sun360_processed.h5 --dataset 0 \
				  --model_path models/sun360/ltla.net --T 4 --M 8 --N 4 \
				  --start_view 2 --save_path dummy/ 
```
- Evaluating SUN360 `ltla` baseline on the test data with `adv` metric:

```
python eval.py --h5_path data/sun360/sun360_processed.h5 --dataset 0 \
				  --model_path models/sun360/ltla.net --T 4 --M 8 --N 4 \
				  --start_view 2 --save_path dummy/ 
```
- Evaluating SUN360 `rnd-actions` baseline on test data with `avg` metric:

```
python eval.py --h5_path data/sun360/sun360_processed.h5 --dataset 0 \
				  --model_path models/sun360/rnd-actions.net --T 4 --M 8 --N 4 \
				  --start_view 2 --actorType random --save_path dummy/ 
```
- Evaluating ModelNet Hard `one-view` baseline on test (seen and unseen) data with `avg` metric:

```
python eval.py --h5_path modelnet30_processed.h5 \
				  --h5_path_unseen modelnet10_processed.h5 --dataset 1 \
				  --model_path models/modelnet_hard/one-view.net --T 1 --M 9 --N 5 \
				  --start_view 2 --save_path dummy/
```

## Training models	
Ensure that the [pre-trained models](http://vision.cs.utexas.edu/projects/sidekicks/models.zip) and [pre-computed scores](http://vision.cs.utexas.edu/projects/sidekicks/scores.zip) are downloaded and extracted. 

- Training `one-view` model on SUN360 with default settings:

```
python main.py --T 1 --training_setting 0 --epochs 100 \
				  --save_path saved_models/sun360/one-view
```
- Training `ltla` baseline on SUN360 with default settings (starting from pre-trained `one-view` model): 

```
python main.py --T 4 --training_setting 1 --epochs 1000 \
				  --save_path saved_models/sun360/ltla/  \
				  --load_model models/sun360/one-view.net
```
- Training `ours-rew` on SUN360 with default settings (with pre-computed score):

```
python main.py --T 4 --training_setting 1 --epochs 1000 \
				  --save_path saved_models/sun360/ours-rew/ \
				  --load_model models/sun360/one-view.net --expert_rewards True \
				  --rewards_h5_path scores/sun360/ours-rew-scores.h5
```
- Training `ours-demo` on SUN360 with default settings (with pre-computed score):

```
python main.py --T 4 --training_setting 1 --epochs 1000 \
				  --save_path saved_models/sun360/ours-demo/ \
				  --load_model models/sun360/one-view.net --expert_trajectories True \
				  --utility_h5_path scores/sun360/ours-demo-scores.h5
```
- Training `ltla` baseline on ModelNet Hard with default settings (starting from pre-trained `one-view` model):

```
python main.py --h5_path data/modelnet_hard/modelnet30_processed.h5 \
				  --training_setting 1 --dataset 1 --T 4 --M 9 --N 5 \
				  --load_model models/modelnet_hard/one-view.net \
				  --save_path saved_models/modelnet_hard/ltla/
```

The other ModelNet Hard models can be trained similar to SUN360 models. To train actor critic models, set `--baselineType critic`. To add full observability to the critic (for `asymm-ac`), set `--critic_full_obs True`. 

## Visualization
From the repository directory, start jupyter notebook and open `visualize_policy_paper.ipynb`. Perform the TODOs mentioned in the comments (setting the correct paths) and run the entire script. It will generate tensorboard files contained visualized heatmaps on several examples. 
