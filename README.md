# SphereFace
A PyTorch Implementation of SphereFace.
The code can be trained on CASIA-Webface and the best accuracy on LFW is **99.22%**.

[SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)

# Train
```
python train.py
```

# Test
```
# lfw.tgz to lfw.zip
tar zxf lfw.tgz; cd lfw; zip -r ../lfw.zip *; cd ..

# lfw evaluation
python lfw_eval.py --model model/sphere20a_20171020.pth
```

# Pre-trained models
| Model name      | LFW accuracy | Training dataset |
|-----------------|--------------|------------------|
| [20171020](model/sphere20a_20171020.7z) | 0.9922 | CASIA-WebFace |

# φ
![equation](https://latex.codecogs.com/gif.latex?phi%28x%29%3D%5Cleft%28-1%5Cright%29%5Ek%5Ccdot%20%5Ccos%20%5Cleft%28x%5Cright%29-2%5Ccdot%20k)

![equation](https://latex.codecogs.com/gif.latex?myphi(x)=1-\frac{x^2}{2!}+\frac{x^4}{4!}-\frac{x^6}{6!}+\frac{x^8}{8!}-\frac{x^9}{9!})

![phi](images/phi.png)

# References
[sphereface](https://github.com/wy1iu/sphereface)