python ./exps/visual.py --meta snapshots/CPM-SBR-vgg11/metas/ --model eval-epoch-022-050-00-01.pth --save cache_data/cache/vgg11
ffmpeg -start_number 3 -i cache_data/cache/vgg11/image%04d.png -b:v 30000k -vf "fps=30" -pix_fmt yuv420p cache_data/cache/vgg11.mp4
cp cache_data/cache/vgg11.mp4 ../.jupyter