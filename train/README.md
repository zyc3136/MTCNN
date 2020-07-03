MTCNN train
==================================

Caffe and Python implementation of Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks.

## Set up

	git clone https://github.com/imistyrain/ssd
	cd ssd
	
**Windows**

	script\build_win.cmd

**Ubuntu**

	mkdir build
	cd build
	cmake ..
	make -j4
	sudo make install

## Prepare data

Download dataset [WIDER](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [FDDB](http://vis-www.cs.umass.edu/fddb/). Put them in data directory like below.

```
data
├── CelebA
│   └── img_celeba
├── fddb
│   ├── FDDB-folds
│   ├── images
│   │   ├── 2002
│   │   └── 2003
│   └── result
│       └── images
└── WIDER
    ├── wider_face_split
    ├── WIDER_test
    ├── WIDER_train
    └── WIDER_val
```

I have write a matlab script to extract WIDER FACE info from matlab `mat` file to `txt` file.

## Train

Prepare data and train network follow the commands in `train.sh`.

## Test

Test the model with `demo.py` for simple detection and `fddb.py` for FDDB benchmark.

## Memory Issue

Since `pNet` may output many bboxes for `rNet` and Caffe's `Blob` never realloc the memory if your new data is smaller, this makes `Blob` only grow the memory and never reduce, which looks like a memory leak. It is fine for most cases but not for our case. You may modify `src/caffe/blob.cpp` if you encounter the memory issue.

```c++
template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  /* some code */
  if (count_ > capacity_) {  // never reduce the memory here
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
}
```

```c++
template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  /* some code */
  if (count_ != capacity_) {  // make a new data buffer
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
}
```