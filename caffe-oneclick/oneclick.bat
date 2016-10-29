echo "Creating train lmdb..."

del "lmdb/train_lmdb\*.*" /f /s /Y
del "lmdb/val_lmdb\*.*" /f /s /Y
rd /s /q "lmdb/train_lmdb"
rd /s /q "lmdb/val_lmdb"

"../bin/convert_imageset" --resize_height=20 --resize_width=20 --shuffle "" "preprocess/train.txt" "lmdb/train_lmdb"

echo "Creating val lmdb..."
"../bin/convert_imageset" --resize_height=20 --resize_width=20 --shuffle "" "preprocess/val.txt" "lmdb/val_lmdb"

echo "computing mean:"

"../bin/compute_image_mean" "lmdb/train_lmdb" "preprocess/mean.binaryproto"

"../bin\caffe.exe" train --solver=modeldef/solver.prototxt

echo "done"
pause