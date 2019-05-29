// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "cpu/vision.h"

// implementation taken from Caffe2
template <typename T>
struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  T w1;
  T w2;
  T w3;
  T w4;
};

// 获得双线性插值采样点周围四个坐标的索引以及对应的权重
template <typename T>
void pre_calc_for_bilinear_interpolate(
    const int height,           // feature map height
    const int width,            // feature map width
    const int pooled_height,    // 7
    const int pooled_width,     // 7
    const int iy_upper,         // roi_bin_grid_h  2
    const int ix_upper,         // roi_bin_grid_w  2
    T roi_start_h,              // y1坐标经过缩放之后的值
    T roi_start_w,              // x1坐标经过缩放之后的值
    T bin_size_h,               // 每个bin的高度
    T bin_size_w,               // 每个bin的宽度
    int roi_bin_grid_h,         // 2 将每个bin划分成2x2大小的grid
    int roi_bin_grid_w,         // 2
    std::vector<PreCalc<T>>& pre_calc   // vector中元素个数为一个roi被划分成的grid数
    ) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {  // 7
    for (int pw = 0; pw < pooled_width; pw++) {  // 7
      for (int iy = 0; iy < iy_upper; iy++) {  // 2
        // roi_start_h是当前roi在特征图上的左上角纵坐标, bin_size_h是每个bin的高度,
        // ph * bin_size_h代表在当前bin之前的所有bin的高度
        // roi_start_h + ph * bin_size_h代表当前bin的左上角纵坐标
        // roi_bin_grid_h=2, bin_size_h / static_cast<T>(roi_bin_grid_h)得到的是bin中每个element的高度
        // 注: 一个bin被划分成了2x2的grid, 也就是4个element
        // 当前bin左上角纵坐标 + 0.5*element的高度 代表 第一行element中心点纵坐标
        // 当前bin左上角纵坐标 + 1.5*element的高度 代表 第二行element中心点纵坐标
        const T yy = roi_start_h + ph * bin_size_h +
            static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5

        for (int ix = 0; ix < ix_upper; ix++) {  // 2
          // 同理, 这里求出的分别是第一列和第二列element中心点横坐标
          const T xx = roi_start_w + pw * bin_size_w +
              static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

          T x = xx;
          T y = yy;
          // deal with: inverse elements are out of feature map boundary
          // TODO: 什么时候会出现这种情况?
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc<T> pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y <= 0) {
            y = 0;
          }
          if (x <= 0) {
            x = 0;
          }

          int y_low = (int)y;
          int x_low = (int)x;
          int y_high;
          int x_high;

          // 图像的双线性插值只会用相邻的4个点, high-low=1
          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (T)y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (T)x_low;
          } else {
            x_high = x_low + 1;
          }

          // 双线性插值, 计算四个点的权重
          // https://blog.csdn.net/u013010889/article/details/79232740
          T ly = y - y_low;
          T lx = x - x_low;
          T hy = 1. - ly, hx = 1. - lx;
          T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indeces
          PreCalc<T> pc;

          // 特征图被拉成了一维数组, 这里直接换算每个element周围的四个点在特征图上的一维坐标
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;

          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;

          pre_calc_index += 1;
        }  //  for: roi_bin_grid_w(2)
      }  // for: roi_bin_grid_h(2)
    }  // for: pooled_width
  }  // for: pooled_height
}

template <typename T>
void ROIAlignForward_cpu_kernel(
    // num_rois * pooled_height * pooled_width * channels
    const int nthreads,
    // feature map (N, C, H, W)
    const T* bottom_data,
    // 1/4 || 1/8 || 1/16 || 1/32
    const T& spatial_scale,
    const int channels,  // C
    const int height,    // H
    const int width,     // W
    const int pooled_height,  // 7
    const int pooled_width,   // 7
    const int sampling_ratio, // 2
    const T* bottom_rois,     // rois.data, 指针类型, 应该指向数组首地址
    //int roi_cols,
    // output.data (num_rois, channels, pooled_height, pooled_width)
    T* top_data
    ) {
  // 这里的T对应到python中指的应该是float型的数

  // 这个指的是传入的每个roi都有5个值, 第一个值是当前img在batch中的索引值
  // 后四个值是坐标
  int roi_cols = 5;

  // roi的数量
  int n_rois = nthreads / channels / pooled_width / pooled_height;

  // (n, c, ph, pw) is an element in the pooled output
  // can be parallelized using omp
  // #pragma omp parallel for num_threads(32)
  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    // 当前(第n个)roi在rois数组中的首地址
    const T* offset_bottom_rois = bottom_rois + n * roi_cols;

    int roi_batch_ind = 0;
    if (roi_cols == 5) {
      // 当前roi所处的image在batch中的索引值
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++;
    }

    // Do not using rounding; this implementation detail is critical
    // 每个roi中的四个坐标值都是相对于原图的坐标, 这里根据当前level特征图的缩放比例\
    // 对坐标进行缩放, 缩放成特征图上的坐标
    T roi_start_w = offset_bottom_rois[0] * spatial_scale;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale;
    // T roi_start_w = round(offset_bottom_rois[0] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[3] * spatial_scale);

    // Force malformed(畸形的) ROIs to be 1x1
    // roi在特征图上的宽度和高度
    T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
    // 将该roi划分成pooled_height*pooled_width大小的grid, 下面是每个bin的大小
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    // 将每个bin划分成2*2大小的grid
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    // we want to precalculate indeces and weights shared by all chanels,
    // this is the key point of optimiation
    // vector中的元素是PreCalc<T>类型, pre_calc中共有
    // roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height个元素
    std::vector<PreCalc<T>> pre_calc(
        roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);

    pre_calc_for_bilinear_interpolate(
        height,
        width,
        pooled_height,
        pooled_width,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        pre_calc);

    for (int c = 0; c < channels; c++) {
      // 当前roi每个channel池化之后特征图的索引初始值(输出)
      int index_n_c = index_n + c * pooled_width * pooled_height;

      // bottom_data是指向特征图的指针, 特征图的维度为 [N, channels, height, width]
      // 其中第一个N指的是当前batch中图片的数量. 此处 roi_batch_ind * channels 表示
      // 当前roi所处的图片在当前特征图上的初始位置, offset_bottom_data则表示当前roi的第
      // c个通道在特征图上的初始位置
      const T* offset_bottom_data =
          bottom_data + (roi_batch_ind * channels + c) * height * width;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ph++) {  // 7
        for (int pw = 0; pw < pooled_width; pw++) {  // 7
          int index = index_n_c + ph * pooled_width + pw;

          T output_val = 0.;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // 2
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {  // 2
              // 下面处理每一个bin
              PreCalc<T> pc = pre_calc[pre_calc_index];

              // offset_bottom_data 指向输入特征图的指针
              output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                  pc.w2 * offset_bottom_data[pc.pos2] +
                  pc.w3 * offset_bottom_data[pc.pos3] +
                  pc.w4 * offset_bottom_data[pc.pos4];

              pre_calc_index += 1;
            }
          }
          output_val /= count;

          // top_data 即 output.data(), 这里应该是把output转换成了一维tensor
          top_data[index] = output_val;
        } // for pw
      } // for ph
    } // for c
  } // for n
}

at::Tensor ROIAlign_forward_cpu(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width,
                                const int sampling_ratio) {
  AT_ASSERTM(!input.type().is_cuda(), "input must be a CPU tensor");
  AT_ASSERTM(!rois.type().is_cuda(), "rois must be a CPU tensor");

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto output = at::empty({num_rois, channels, pooled_height, pooled_width}, input.options());
  auto output_size = num_rois * pooled_height * pooled_width * channels;

  if (output.numel() == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input.type(), "ROIAlign_forward", [&] {
    ROIAlignForward_cpu_kernel<scalar_t>(
         output_size,
         input.data<scalar_t>(),
         spatial_scale,
         channels,
         height,
         width,
         pooled_height,
         pooled_width,
         sampling_ratio,
         rois.data<scalar_t>(),
         output.data<scalar_t>());
  });
  return output;
}
