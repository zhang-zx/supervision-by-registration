python ./exps/visual.py --begin snapshots/test/metas/eval-begin-eval-00-02.pth --last snapshots/test/metas/eval-last-eval-00-02.pth --save cache_data/cache/test
ffmpeg -start_number 3 -i cache_data/cache/test/image%04d.png -b:v 30000k -vf "fps=30" -pix_fmt yuv420p cache_data/cache/test.mp4
cp cache_data/cache/test.mp4 ../.jupyter