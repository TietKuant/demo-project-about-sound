# Thử nghiệm mô hình phân loại Audio Router V4 MVP

## Mục tiêu thử nghiệm

Thử nghiệm này nhằm đánh giá một mô hình phân loại âm thanh nhẹ cho tầng Audio Router V4. Mục tiêu của router là nhận diện loại nội dung âm thanh đầu vào và đề xuất hướng xử lý phù hợp trong hệ thống. Đây là phiên bản MVP gồm 4 lớp, chưa phải bộ định tuyến hoàn chỉnh cho toàn bộ hệ thống trong tương lai.

Bốn lớp được sử dụng trong thử nghiệm gồm:

- `environment_only`: âm thanh môi trường không chứa tiếng nói hoặc nhạc.
- `music_with_vocals`: nhạc có giọng hát.
- `speech_clean`: tiếng nói sạch, không cần xử lý khử nhiễu.
- `speech_target_noise`: tiếng nói có lẫn các loại nhiễu mục tiêu mà đề tài hỗ trợ.

Lớp `speech_noisy_general` chưa được đưa vào MVP vì dữ liệu hiện tại không có phân chia train/test hợp lệ cho lớp này. Nếu đưa lớp này vào đánh giá, kết quả sẽ không phản ánh đúng năng lực học của mô hình vì mô hình không có dữ liệu huấn luyện tương ứng.

## Thiết kế dữ liệu

Bộ manifest cuối cùng có tổng cộng 2144 mẫu, gồm 1700 mẫu huấn luyện và 444 mẫu kiểm thử. Dữ liệu được tổng hợp từ nhiều nguồn khác nhau:

| Nguồn dữ liệu | Số mẫu |
|---|---:|
| ESC-50 | 600 |
| MUSDB18 Preview | 144 |
| Target-noise v2 source-disjoint | 400 |
| UrbanSound8K | 600 |
| VoiceBank clean | 400 |

Phân bố theo nhãn nội dung:

| Nhãn nội dung | Số mẫu |
|---|---:|
| environment_only | 1200 |
| music_with_vocals | 144 |
| speech_clean | 400 |
| speech_target_noise | 400 |

Bộ dữ liệu target-noise sử dụng phiên bản `target_noise_v2_source_disjoint`, trong đó các nguồn clean speech và noise source được kiểm soát để không bị trùng giữa train và test. Điều này giúp giảm nguy cơ mô hình học thuộc nguồn dữ liệu thay vì học đặc trưng âm thanh thật.

## Ánh xạ nhãn sang hướng xử lý

Mỗi nhãn nội dung được ánh xạ sang một route xử lý và engine tương ứng:

| Nhãn nội dung | Route target | Engine target |
|---|---|---|
| environment_only | out_of_scope | none |
| music_with_vocals | manual_required | demucs |
| speech_clean | no_process | none |
| speech_target_noise | target_noise_suppression | target_noise_suppressor |

Cách ánh xạ này giúp tách rõ hai khái niệm: router là mô hình phân loại nội dung âm thanh, còn processor/engine là thành phần thực hiện xử lý tín hiệu âm thanh.

## Phương pháp huấn luyện

Mô hình sử dụng các đặc trưng âm thanh nhẹ như năng lượng RMS, zero-crossing rate, spectral centroid, spectral bandwidth, spectral rolloff, spectral flatness, tỉ lệ năng lượng theo dải tần, độ biến thiên RMS/ZCR và silence ratio. Trường `duration_sec` được giữ lại để kiểm tra chất lượng dữ liệu nhưng không đưa vào huấn luyện nhằm giảm nguy cơ mô hình học theo dấu vết nguồn dữ liệu.

Sau baseline ban đầu, thử nghiệm được cải tiến bằng hàm mất mát có trọng số theo lớp (`class_weighting=balanced`). Trọng số lớp được tính từ tập train theo công thức:

`total_train_rows / (num_classes * train_count_for_class)`

Trọng số lớp của bản cải tiến:

| Lớp | Trọng số |
|---|---:|
| environment_only | 0.4187 |
| music_with_vocals | 4.5213 |
| speech_clean | 1.1273 |
| speech_target_noise | 1.9860 |

## Kết quả đánh giá chính

Kết quả chính dưới đây sử dụng bản `class_weighting=balanced`:

| Chỉ số | Giá trị |
|---|---:|
| Accuracy | 0.8739 |
| Macro-F1 | 0.8597 |
| Weighted-F1 | 0.8763 |

Kết quả theo từng lớp:

| Lớp | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| environment_only | 0.9290 | 0.8486 | 0.8870 | 185 |
| music_with_vocals | 0.9434 | 1.0000 | 0.9709 | 50 |
| speech_clean | 0.5789 | 0.9565 | 0.7213 | 23 |
| speech_target_noise | 0.8641 | 0.8548 | 0.8595 | 186 |

Kết quả theo nguồn dữ liệu:

| Nguồn dữ liệu | Số mẫu test | Accuracy |
|---|---:|---:|
| esc50 | 126 | 0.8492 |
| musdb18_preview | 50 | 1.0000 |
| target_noise_v2_source_disjoint | 186 | 0.8548 |
| urbansound8k | 59 | 0.8475 |
| voicebank_clean | 23 | 0.9565 |

## So sánh với baseline không trọng số

Bản baseline ban đầu dùng cross-entropy không trọng số. Bản cải tiến dùng balanced loss để giảm hiện tượng mô hình bỏ sót lớp `speech_target_noise`.

| Phiên bản | Accuracy | Macro-F1 | Target-noise Precision | Target-noise Recall | Target-noise F1 |
|---|---:|---:|---:|---:|---:|
| Không trọng số | 0.7905 | 0.7792 | 0.9545 | 0.5645 | 0.7095 |
| Balanced loss | 0.8739 | 0.8597 | 0.8641 | 0.8548 | 0.8595 |

Ma trận lỗi của lớp `speech_target_noise` được cải thiện rõ rệt:

| Phiên bản | Dự đoán đúng target-noise | Nhầm sang environment_only | Nhầm sang speech_clean | Nhầm sang music_with_vocals |
|---|---:|---:|---:|---:|
| Không trọng số | 105 | 58 | 23 | 0 |
| Balanced loss | 159 | 12 | 14 | 1 |

## Nhận xét

Balanced loss cải thiện đáng kể điểm yếu chính của baseline ban đầu. Recall của lớp `speech_target_noise` tăng từ 0.5645 lên 0.8548, trong khi F1 của lớp này tăng từ 0.7095 lên 0.8595. Số mẫu target-noise bị nhầm thành `environment_only` giảm từ 58 xuống 12.

Precision của `speech_target_noise` giảm từ 0.9545 xuống 0.8641, nhưng đây là đánh đổi hợp lý vì macro-F1 tổng thể cũng tăng từ 0.7792 lên 0.8597. Điều này cho thấy cải tiến không chỉ làm tăng recall của một lớp riêng lẻ mà còn tạo ra baseline router tốt hơn về tổng thể.

## Giới hạn của thử nghiệm

Thử nghiệm này có ba giới hạn chính. Thứ nhất, đây là router MVP 4 lớp, chưa bao gồm toàn bộ taxonomy tương lai. Thứ hai, lớp `speech_noisy_general` bị loại khỏi MVP vì dữ liệu hiện tại chưa có train coverage hợp lệ. Thứ ba, mô hình vẫn dựa trên đặc trưng âm thanh nhẹ, chưa sử dụng pretrained audio embeddings hoặc mô hình âm thanh lớn hơn.

Do đó, kết quả này nên được mô tả là một improved lightweight-feature baseline cho Audio Router V4 MVP. Hệ thống demo hiện tại không nên được mô tả là một ML router hoàn chỉnh nếu mô hình này chưa được tích hợp trực tiếp vào app/analyzer.
