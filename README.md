# SphereFace
A PyTorch Implementation of SphereFace.
The code can be trained on CASIA-Webface and the best accuracy on LFW is **99.15%**.

[SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)

# Usage
```
python train.py
```
# φ
![equation](https://latex.codecogs.com/gif.latex?phi%28x%29%3D%5Cleft%28-1%5Cright%29%5Ek%5Ccdot%20%5Ccos%20%5Cleft%28x%5Cright%29-2%5Ccdot%20k)

![equation](https://latex.codecogs.com/gif.latex?myphi(x)=1-\frac{x^2}{2!}+\frac{x^4}{4!}-\frac{x^6}{6!}+\frac{x^8}{8!}-\frac{x^9}{9!})

![phi](images/phi.png)
