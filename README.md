# 🚀 Tối ưu thuật toán Sobel phát hiện biên ảnh bằng OpenCL

Dự án này gồm:
- Chạy thuật toán Sobel trên CPU và GPU
- Sau đó đánh giá so sánh về thời gian, kết quả ảnh sau khi phát hiện biên để xem cái nào hiệu quả hơn

## CPU thực hiện Sobel

- Đầu tiên CPU thực hiện Sobel bằng OpenCV:

```
cv::filter2D(input, grad_x, CV_16S, kernelX);
cv::filter2D(input, grad_y, CV_16S, kernelY);
```
kernelX và kernelY chính là “Sobel X” và “Sobel Y” của ma trận Sobel

- Sau đó:
```
cv::convertScaleAbs(...)
cv::addWeighted(...)
```
CPU tính biên độ (magnitude) và xuất ảnh Sobel

## GPU thực hiện Sobel

- GPU chạy file kernel chính là edge_filter.cl: 
```
sumX += pixel * Gx[..];
sumY += pixel * Gy[..];
magnitude = sqrt(sumX*sumX + sumY*sumY);
```
GPU sẽ tính Sobel từng pixel một
