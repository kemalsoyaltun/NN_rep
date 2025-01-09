#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <cstdint>
#include <limits>
#include <opencv2/opencv.hpp> // OpenCV başlık dosyasını ekleyin

std::vector<unsigned char> readUByteFile(const std::string& filePath) {
    std::vector<unsigned char> data;
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);

    if (!file) {
        std::cerr << "Dosya açılamadı: " << filePath << std::endl;
        return data; // Boş vektör döner
    }

    std::cerr << "Dosya başarıyla açıldı: " << filePath << std::endl;

    file.clear(); // Hata bayraklarını temizle
    file.seekg(0, std::ios::end); // Dosya sonuna git
    std::streampos fileSize = file.tellg(); // Dosya boyutunu al
    file.seekg(0, std::ios::beg); // Dosya başına dön

    if (fileSize <= 0 || fileSize > std::numeric_limits<std::streamsize>::max()) {
        std::cerr << "Dosya boyutu geçersiz: " << fileSize << " bayt." << std::endl;
        return data;
    }

    // std::cerr << "Dosya boyutu: " << fileSize << " bayt." << std::endl;

    file.unsetf(std::ios::skipws);
    data.reserve(static_cast<size_t>(fileSize)); // Bellekte yer ayır
    data.insert(data.begin(),
                std::istream_iterator<unsigned char>(file),
                std::istream_iterator<unsigned char>());

    if (data.empty()) {
        std::cerr << "Veri boş: Dosya içeriği okunamadı veya dosya boş." << std::endl;
    } else {
        // std::cerr << "Dosyadan " << data.size() << " bayt okundu." << std::endl;
    }

    return data;
}

// int main() {
//     // Görüntü dosyasını oku
//     std::string imageFilePath = "/home/yorgundemokrat/cpptr/Dataset/train-images.idx3-ubyte"; 
//     std::vector<unsigned char> imageData = readUByteFile(imageFilePath);

//     // Etiket dosyasını oku
//     std::string labelFilePath = "/home/yorgundemokrat/cpptr/Dataset/train-labels.idx1-ubyte";
//     std::vector<unsigned char> labelData = readUByteFile(labelFilePath);

//     if (!imageData.empty() && !labelData.empty()) {
//         // Görüntü başlığından verileri çıkarma
//         int magicNumber = (imageData[0] << 24) | (imageData[1] << 16) | (imageData[2] << 8) | imageData[3];
//         int numberOfImages = (imageData[4] << 24) | (imageData[5] << 16) | (imageData[6] << 8) | imageData[7];
//         int numberOfRows = (imageData[8] << 24) | (imageData[9] << 16) | (imageData[10] << 8) | imageData[11];
//         int numberOfColumns = (imageData[12] << 24) | (imageData[13] << 16) | (imageData[14] << 8) | imageData[15];
        
//         std::cout << "Sihirli Sayı: " << magicNumber << std::endl;
//         std::cout << "Görüntü Sayısı: " << numberOfImages << std::endl;
//         std::cout << "Görüntü Boyutu: " << numberOfRows << "x" << numberOfColumns << std::endl;
        
//         // Görüntü verisinin boyutu
//         int imageSize = numberOfRows * numberOfColumns;
        
//         // İlk görüntünün verisini okuma
//         std::vector<unsigned char> firstImage(imageData.begin() + 16, imageData.begin() + 16 + imageSize);

//         // İlk görüntünün etiketini okuma
//         int firstLabel = labelData[8]; // İlk 8 bayt başlık, bu yüzden 8. bayttan başlıyoruz

//         std::cout << "İlk Görüntünün Etiketi: " << firstLabel << std::endl;
        
//         // Görüntüyü OpenCV Mat formatına çevirme
//         cv::Mat image(numberOfRows, numberOfColumns, CV_8UC1, firstImage.data());

//         std::cout << "Image Size: " << image.size() << std::endl;

//         for (int i = 0; i < firstImage.size(); ++i) {
//             if (firstImage[i] > 255) {
//                 std::cerr << "Veri formatı hatası: Piksel değeri 255'ten büyük." << std::endl;
//                 break;
//             }
//         }

//         std::string windowName = "İlk Görüntü";
//         cv::namedWindow(windowName, cv::WINDOW_NORMAL);

//         // Görüntüyü gösterme
//         cv::imshow(windowName, image);
//         cv::waitKey(0); // Kullanıcı bir tuşa basana kadar bekle

//         // Görüntüyü kaydetmek isterseniz:
//         // cv::imwrite("ilk_goruntu.png", image);

//     } else {
//         std::cerr << "Görüntü veya etiket verisi yüklenemedi." << std::endl;
//     }

//     return 0;
// }