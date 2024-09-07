# FaceRestorer

人脸修复算法合集，集成了[GFPGAN](https://github.com/TencentARC/GFPGAN)、[CodeFormer](https://github.com/sczhou/CodeFormer)、[RestoreFormer](https://github.com/wzhouxiff/https://github.com/wzhouxiff/RestoreFormer)、[RestoreFormerPlusPlus](https://github.com/wzhouxiff/RestoreFormerPlusPlus)，以及背景放大的[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)，并且对于使用了已弃用函数的代码进行了更新。

## 使用

1. 克隆本项目

   ```bash
   git clone https://github.com/ZuoguanPikachu/FaceRestorer.git
   ```

2. 从Release中下载相应的模型权重到`face_restorer/models/`下相应的文件夹中。