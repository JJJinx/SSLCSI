This website provides dataset splits and the code for SSLCSI.

## Abstract
Recently, with the advancement of the Internet of Things (IoT), WiFi CSI-based HAR has gained increasing attention from academic and industry communities.
By integrating the deep learning technology with CSI-based HAR, researchers achieve state-of-the-art performance without the need of expert knowledge.
However, the scarcity of labeled CSI data remains the most prominent challenge when applying deep learning models in the context of CSI-based HAR due to the privacy and incomprehensibility of CSI-based HAR data.
On the other hand, SSL has emerged as a promising approach for learning meaningful representations from data without heavy reliance on labeled examples.
Therefore, considerable efforts have been made to address the challenge of insufficient data in deep learning by leveraging SSL algorithms.
In this paper, we undertake a comprehensive inventory and analysis of the potential held by different categories of SSL algorithms, including those that have been previously studied and those that have not yet been explored, within the field.
We provide an in-depth investigation of SSL algorithms in the context of WiFi CSI-based HAR.
We evaluate four categories of SSL algorithms using three publicly available CSI HAR datasets, each encompassing different tasks and environmental settings.
To ensure relevance to real-world applications, we design performance metrics that align with specific requirements.
Furthermore, our experimental findings uncover several limitations and blind spots in existing work, highlighting the barriers that need to be addressed before SSL can be effectively deployed in real-world WiFi-based HAR applications. Our results also serve as a practical guideline for industry practitioners and provide valuable insights for future research endeavors in this field.

## Datasets and Splits

| Dataset | Description | URL |
| ----- | ----------- | ---- |
| UT-HAR | It consists of 557 recordings collected from six participants engaging in six distinct coarse-grained activities: lying down, falling, picking up an object, running, sitting down, standing up, and walking. | [Link](https://github.com/ermongroup/Wifi_Activity_Recognition) |
| Falldefi | It contains 553 records of fall or other actions (not fall). The data was recorded in 3 environments.| [Link](https://github.com/dmsp123/FallDeFi) |
| Signfi | It comprises recordings of 276 sign gestures performed by 5 participants in two distinct environments: lab and home. The activities were captured using a receiver equipped with an Intel 5300 NIC and 3 antennas, and each recording includes data from 30 subcarriers.  To account for environmental variability, we used a subset of the dataset that specifically includes 2760 records recorded in the home environment.  Each gesture in the dataset is represented by 10 records. We selected this dataset because it encompasses nearly all of the 300 most commonly used basic sign gestures in daily life. | [Link](https://yongsen.github.io/SignFi/) |
| Widar | It consists of recordings from 17 users performing 22 gesture activities in three different rooms. The dataset utilizes one transmitter and six receivers, each equipped with an Intel 5300 NIC and 3 antennas, placed at different locations. Each activity was simultaneously captured on all six receivers using 30 subcarriers, enabling the collection of multiple perspectives of the same activity. Considering each receiver's data as separate, the Widar dataset gathers a total of 271,038 data points. | [Link](http://tns.thss.tsinghua.edu.cn/widar3.0/) |

## Task Generalizability of the Representations. 


<table>
    <tr>
        <td>Classifer</td>
        <td colspan="4">1-layer FC</td>
        <td colspan="4">2-layer FC</td>
    <tr>
    <tr>
        <td>Dataset</td>
        <td>UT-HAR</td>
        <td>Falldefi</td>
        <td>SignFi</td>
        <td>Widar\_R2</td>
        <td>UT-HAR</td>
        <td>Falldefi</td>
        <td>SignFi</td>
        <td>Widar\_R2</td>
    <tr>
    <tr>
        <td>Supervised</td>
        <td>98.03</td>
        <td>91.07</td>
        <td>97.65</td>
        <td>94.37</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    <tr>
    <tr>
        <td>SimCLR</td>
        <td>83.93</td>
        <td>77.68</td>
        <td>47.10</td>
        <td>40.43</td>
        <td>94.64</td>
        <td>88.39</td>
        <td>51.81</td>
        <td>57.58</td>
    <tr>
    <tr>
        <td>MoCo</td>
        <td>82.50</td>
        <td>77.68</td>
        <td>62.64</td>
        <td>41.13</td>
        <td>92.86</td>
        <td>90.18</td>
        <td>64.49</td>
        <td>52.80</td>
    <tr>
    <tr>
        <td>SwAV</td>
        <td>87.50</td>
        <td>77.68</td>
        <td>58.41</td>
        <td>36.39</td>
        <td>94.64</td>
        <td>86.60</td>
        <td>65.58</td>
        <td>55.26</td>
    <tr>
    <tr>
        <td>Rel-Pos</td>
        <td>82.14</td>
        <td>74.11</td>
        <td>81.34</td>
        <td>32.18</td>
        <td>91.07</td>
        <td>82.14</td>
        <td>92.03</td>
        <td>48.59</td>
    <tr>
    <tr>
        <td>MoCo (ViT)</td>
        <td>73.21</td>
        <td>74.11</td>
        <td>87.14</td>
        <td>62.81</td>
        <td>91.07</td>
        <td>79.46</td>
        <td>97.46</td>
        <td>56.04</td>
    <tr>
    <tr>
        <td>MAE (ViT)</td>
        <td>84.29</td>
        <td>78.57</td>
        <td>88.77</td>
        <td>69.24</td>
        <td>94.64</td>
        <td>85.71</td>
        <td>94.20</td>
        <td>89.14</td>
    <tr>
</table>





## Robustness to Domain Shift.







## Robustness of SSL Algorithms to Data Scarcity.







## Cite the Paper
> Xu, K., Wang, J., Zhu, H. and Zheng, D., 2023. Self-Supervised Learning for WiFi CSI-Based Human Activity Recognition: A Systematic Study. arXiv preprint arXiv:2308.02412.

