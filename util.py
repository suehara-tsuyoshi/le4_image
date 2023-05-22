import numpy as np 

def im2col(input_data, filter_h, filter_w, stride=1, pad=0) :
	# Parameters
  #   ----------
  #   input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
  #   filter_h : フィルターの高さ
  #   filter_w : フィルターの幅
  #   stride : ストライド
  #   pad : パディング
  #   Returns
  #   -------
  #   col : 2次元配列
	
