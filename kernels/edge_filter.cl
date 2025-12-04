__kernel void edge_filter(__global const uchar *input, __global uchar *output,
                          const int width, const int height) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) {
    output[y * width + x] = 0;
    return;
  }

  int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

  int Gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

  int sumX = 0;
  int sumY = 0;

  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      int px = x + dx;
      int py = y + dy;
      uchar pixel = input[py * width + px];
      sumX += pixel * Gx[dy + 1][dx + 1];
      sumY += pixel * Gy[dy + 1][dx + 1];
    }
  }

  int magnitude = (int)sqrt((float)(sumX * sumX + sumY * sumY));
  if (magnitude > 255)
    magnitude = 255;
  output[y * width + x] = (uchar)magnitude;
}
