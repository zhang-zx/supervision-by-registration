python ./exps/visual.py --begin snapshots/test/metas/eval-begin-eval-01-02.pth --last snapshots/test/metas/eval-last-eval-01-02.pth --save cache_data/cache/sbr-mobile-demo
ffmpeg -start_number 3 -i cache_data/cache/sbr-mobile-demo/image%05d.png -b:v 30000k -vf "fps=30" -pix_fmt yuv420p cache_data/cache/sbr-mobile-demo.mp4
cp cache_data/cache/sbr-mobile-demo.mp4 ../.jupyter
