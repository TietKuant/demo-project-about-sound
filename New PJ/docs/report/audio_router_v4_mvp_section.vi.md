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

## Kết quả đánh giá

Kết quả đánh giá trên tập test như sau:

| Chỉ số | Giá trị |
|---|---:|
| Accuracy | 0.7905 |
| Macro-F1 | 0.7792 |
| Weighted-F1 | 0.7862 |

Kết quả theo từng lớp:

| Lớp | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| environment_only | 0.7427 | 0.9676 | 0.8404 | 185 |
| music_with_vocals | 1.0000 | 0.9000 | 0.9474 | 50 |
| speech_clean | 0.4583 | 0.9565 | 0.6197 | 23 |
| speech_target_noise | 0.9545 | 0.5645 | 0.7095 | 186 |

Kết quả theo nguồn dữ liệu:

| Nguồn dữ liệu | Số mẫu test | Accuracy |
|---|---:|---:|
| ESC-50 | 126 | 0.9603 |
| MUSDB18 Preview | 50 | 0.9000 |
| Target-noise v2 source-disjoint | 186 | 0.5645 |
| UrbanSound8K | 59 | 0.9831 |
| VoiceBank clean | 23 | 0.9565 |

## Nhận xét

Mô hình đạt kết quả tốt trên các lớp `environment_only` và `music_with_vocals`, cho thấy các đặc trưng âm thanh nhẹ có thể phân biệt tương đối tốt giữa âm thanh môi trường và nhạc có giọng hát. Lớp `speech_clean` có recall cao nhưng precision thấp hơn, nghĩa là mô hình có xu hướng dự đoán một số mẫu khác thành tiếng nói sạch.

Đối với lớp `speech_target_noise`, precision đạt 0.9545 nhưng recall chỉ đạt 0.5645. Điều này cho thấy khi mô hình dự đoán một mẫu là target-noise thì thường đúng, nhưng mô hình vẫn bỏ sót nhiều mẫu target-noise. Vì vậy, mô hình hiện tại phù hợp để xem là baseline có kiểm soát, chưa đủ để xem là router tự động cấp production.

## Giới hạn của thử nghiệm

Thử nghiệm này có ba giới hạn chính. Thứ nhất, đây là router MVP 4 lớp, chưa bao gồm toàn bộ taxonomy tương lai. Thứ hai, lớp `speech_noisy_general` bị loại khỏi MVP vì dữ liệu hiện tại chưa có train coverage hợp lệ. Thứ ba, mô hình sử dụng đặc trưng âm thanh nhẹ nên khả năng nhận diện target-noise còn hạn chế, đặc biệt ở recall.

Do đó, hệ thống demo hiện tại không nên được mô tả là một ML router hoàn chỉnh nếu mô hình này chưa được tích hợp trực tiếp vào app/analyzer. Kết quả này nên được trình bày là một baseline thử nghiệm cho hướng xây dựng router học máy trong hệ thống xử lý âm thanh.
