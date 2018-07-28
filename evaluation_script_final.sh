data_path=data/sun360/sun360_processed.h5
model_path=/projects/vision/1/webspace/projects/sidekicks/models/sun360

echo "=========================================================================="
echo "=========================== SUN360 results ==============================="
echo "=========================================================================="

echo "===> Evaluating one-view"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --dataset 0 \
			         	 --model_path $model_path/one-view.net --T 1 --M 8 --N 4 \
			   			 --start_view 2 --save_path dummy/ 

echo "===> Evaluating rnd-actions"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --dataset 0 \
			         	 --model_path $model_path/rnd-actions.net --T 4 --M 8 --N 4 \
			   			 --start_view 2 --save_path dummy/  --actorType random

echo "===> Evaluating rnd-rewards"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --dataset 0 \
			         	 --model_path $model_path/rnd-rewards.net --T 4 --M 8 --N 4 \
			   			 --start_view 2 --save_path dummy/  
echo "===> Evaluating ltla"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --dataset 0 \
			         	 --model_path $model_path/ltla.net --T 4 --M 8 --N 4 \
			   			 --start_view 2 --save_path dummy/ 

echo "===> Evaluating asymm-ac"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --dataset 0 \
			         	 --model_path $model_path/asymm-ac.net --T 4 --M 8 --N 4 \
			   			 --start_view 2 --save_path dummy/ --baselineType critic \
						 --critic_full_obs True 

echo "===> Evaluating expert-clone"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --dataset 0 \
			         	 --model_path $model_path/expert-clone.net --T 4 --M 8 --N 4 \
			   			 --start_view 2 --save_path dummy/ 

echo "===> Evaluating ours(rew)"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --dataset 0 \
			         	 --model_path $model_path/ours-rew.net --T 4 --M 8 --N 4 \
			   			 --start_view 2 --save_path dummy/ 

echo "===> Evaluating ours(demo)"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --dataset 0 \
			         	 --model_path $model_path/ours-demo.net --T 4 --M 8 --N 4 \
			   			 --start_view 2 --save_path dummy/ 

echo "===> Evaluating ours(rew)+ac"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --dataset 0 \
			         	 --model_path $model_path/ours-rew-ac.net --T 4 --M 8 --N 4 \
			   			 --start_view 2 --save_path dummy/ --baselineType critic

echo "===> Evaluating ours(demo)+ac"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --dataset 0 \
			         	 --model_path $model_path/ours-demo-ac.net --T 4 --M 8 --N 4 \
			   			 --start_view 2 --save_path dummy/ --baselineType critic


echo "===> Evaluating demo-actions"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --dataset 0 \
			         	 --model_path $model_path/demo-actions.net --T 4 --M 8 --N 4 \
			   			 --start_view 2 --save_path dummy/ --actorType demo_sidekick \
						 --utility_h5_path scores/sun360/ours-demo-scores.h5

data_path=data/modelnet_hard/modelnet30_processed.h5
data_path_unseen=data/modelnet_hard/modelnet10_processed.h5
model_path=/projects/vision/1/webspace/projects/sidekicks/models/modelnet_hard

echo "=========================================================================="
echo "======================== ModelNet Hard results ==========================="
echo "=========================================================================="

echo "===> Evaluating one-view"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --h5_path_unseen $data_path_unseen --dataset 1 \
			         	 --model_path $model_path/one-view.net --T 1 --M 9 --N 5 \
			   			 --start_view 2 --save_path dummy/ 

echo "===> Evaluating rnd-actions"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --h5_path_unseen $data_path_unseen --dataset 1 \
			         	 --model_path $model_path/rnd-actions.net --T 4 --M 9 --N 5 \
			   			 --start_view 2 --save_path dummy/  --actorType random

echo "===> Evaluating rnd-rewards"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --h5_path_unseen $data_path_unseen --dataset 1 \
			         	 --model_path $model_path/rnd-rewards.net --T 4 --M 9 --N 5 \
			   			 --start_view 2 --save_path dummy/  
echo "===> Evaluating ltla"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --h5_path_unseen $data_path_unseen --dataset 1 \
			         	 --model_path $model_path/ltla.net --T 4 --M 9 --N 5 \
			   			 --start_view 2 --save_path dummy/ 

echo "===> Evaluating asymm-ac"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --h5_path_unseen $data_path_unseen --dataset 1 \
			         	 --model_path $model_path/asymm-ac.net --T 4 --M 9 --N 5 \
			   			 --start_view 2 --save_path dummy/ --baselineType critic \
						 --critic_full_obs True 

echo "===> Evaluating expert-clone"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --h5_path_unseen $data_path_unseen --dataset 1 \
			         	 --model_path $model_path/expert-clone.net --T 4 --M 9 --N 5 \
			   			 --start_view 2 --save_path dummy/ 

echo "===> Evaluating ours(rew)"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --h5_path_unseen $data_path_unseen --dataset 1 \
			         	 --model_path $model_path/ours-rew.net --T 4 --M 9 --N 5 \
			   			 --start_view 2 --save_path dummy/ 

echo "===> Evaluating ours(demo)"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --h5_path_unseen $data_path_unseen --dataset 1 \
			         	 --model_path $model_path/ours-demo.net --T 4 --M 9 --N 5 \
			   			 --start_view 2 --save_path dummy/ 

echo "===> Evaluating ours(rew)+ac"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --h5_path_unseen $data_path_unseen --dataset 1 \
			         	 --model_path $model_path/ours-rew-ac.net --T 4 --M 9 --N 5 \
			   			 --start_view 2 --save_path dummy/ --baselineType critic

echo "===> Evaluating ours(demo)+ac"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --h5_path_unseen $data_path_unseen --dataset 1 \
			         	 --model_path $model_path/ours-demo-ac.net --T 4 --M 9 --N 5 \
			   			 --start_view 2 --save_path dummy/ --baselineType critic


echo "===> Evaluating demo-actions"
echo "====== Average score ======"
python -W ignore eval.py --h5_path $data_path --h5_path_unseen $data_path_unseen --dataset 1 \
			         	 --model_path $model_path/demo-actions.net --T 4 --M 9 --N 5 \
			   			 --start_view 2 --save_path dummy/ --actorType demo_sidekick \
						 --utility_h5_path scores/modelnet_hard/ours-demo-scores.h5
