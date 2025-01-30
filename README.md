Our paper has been accepted. In the coming weeks, we will organize and upload the data splits and code, along with additional results that were not included in the paper.
For those who have fill the [request form](https://docs.google.com/forms/d/e/1FAIpQLScgXk6Ok33BL4S49cVRtQ-65mZu1Q1qZHgqFvtNEmCUBCfniA/viewform?usp=sf_link), we will send you an notification email.

# Todo List
- [ ] Present the results of the paper.
- [ ] Experiment on the influence of different preprocessing methods on CSI data.
- [ ] Conduct a benchmark evaluation of various SSL algorithms using datasets provided by [SDP](http://www.sdp8.org/).


## Cite the Paper
> Xu, K., Wang, J., Zhu, H. and Zheng, D., 2023. Self-Supervised Learning for WiFi CSI-Based Human Activity Recognition: A Systematic Study. arXiv preprint arXiv:2308.02412.
> @article{10.1145/3715130,
author = {Xu, Ke and Wang, Jiangtao and Zhu, Hongyuan and Zheng, Dingchang},
title = {Evaluating Self-Supervised Learning for WiFi CSI-Based Human Activity Recognition},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1550-4859},
url = {https://doi.org/10.1145/3715130},
doi = {10.1145/3715130},
abstract = {With the advancement of the Internet of Things (IoT), WiFi Channel State Information (CSI)-based Human Activity Recognition (HAR) has garnered increasing attention from both academic and industrial communities. However, the scarcity of labeled data remains a prominent challenge in CSI-based HAR, primarily due to privacy concerns and the incomprehensibility of CSI data. Concurrently, Self-Supervised Learning (SSL) has emerged as a promising approach for addressing the dilemma of insufficient labeled data. In this paper, we undertake a comprehensive inventory and analysis of different categories of SSL algorithms, encompassing both previously studied and unexplored approaches within the field. We provide an in-depth investigation and evaluation of SSL algorithms in the context of WiFi CSI-based HAR, utilizing publicly available datasets that encompass various tasks and environmental settings. To ensure relevance to real-world applications, we design experiment settings aligned with specific requirements. Furthermore, our experimental findings uncover several limitations and blind spots in existing work, shedding light on the barriers that need to be addressed before SSL can be effectively deployed in real-world WiFi-based HAR applications. Our results also serve as practical guidelines and provide valuable insights for future research endeavors in this field.},
note = {Just Accepted},
journal = {ACM Trans. Sen. Netw.},
month = jan,
keywords = {WiFi, Channel State Information, Self-Supervised Learning, Human Activity Recognition}
}

## Datasets and Splits

| Dataset | Description | URL |
| ----- | ----------- | ---- |
| UT-HAR | It consists of 557 recordings collected from six participants engaging in six distinct coarse-grained activities: lying down, falling, picking up an object, running, sitting down, standing up, and walking. | [Link](https://github.com/ermongroup/Wifi_Activity_Recognition) |
| Falldefi | It contains 553 records of fall or other actions (not fall). The data was recorded in 3 environments.| [Link](https://github.com/dmsp123/FallDeFi) |
| Signfi | It comprises recordings of 276 sign gestures performed by 5 participants in two distinct environments: lab and home. The activities were captured using a receiver equipped with an Intel 5300 NIC and 3 antennas, and each recording includes data from 30 subcarriers.  To account for environmental variability, we used a subset of the dataset that specifically includes 2760 records recorded in the home environment.  Each gesture in the dataset is represented by 10 records. We selected this dataset because it encompasses nearly all of the 300 most commonly used basic sign gestures in daily life. | [Link](https://yongsen.github.io/SignFi/) |
| Widar | It consists of recordings from 17 users performing 22 gesture activities in three different rooms. The dataset utilizes one transmitter and six receivers, each equipped with an Intel 5300 NIC and 3 antennas, placed at different locations. Each activity was simultaneously captured on all six receivers using 30 subcarriers, enabling the collection of multiple perspectives of the same activity. Considering each receiver's data as separate, the Widar dataset gathers a total of 271,038 data points. | [Link](http://tns.thss.tsinghua.edu.cn/widar3.0/) |




