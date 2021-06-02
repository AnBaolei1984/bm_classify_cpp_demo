# bm_classify_cpp_demo
这是在比特大陆BM1684 AI硬件上运行分类网络的demo

1) 编译

    a) 如果硬件是SC5/SC5+ 板卡系列,
    
    在x86机器上，
    
      make -f Makefile.pcie -j4
    
    在飞腾类arm服务器上
    
      make -f Makefile.arm_pcie -j4
    
    b) 如果硬件是SE5/SM5等SoC设备,

      make -f Makefile.arm -j4
      
      编译成功后将生成的可执行文件classify_test拷贝到SE5/SM5上执行。
    
2) 运行

    ./classify_test 1620470040619267.mp4 resnet18_4_int8.bmodel

