# 车道线检测 Kalman Filter

## 原理

预测过程 $s$

$$
\begin{equation}
    x_{k+1} = F_k x_k + B_k u_k + w_k
\end{equation}
$$