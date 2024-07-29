# PPFedGNN
Exploring the intrinsic value of social data has long been a focal point for researchers. Presently, diverse social network data is dispersed across various platforms. While federated learning enables collaborative training with multiple clients, enhancing model performance while safeguarding client-specific information, it often overlooks global user relevance, inconsistent distribution of social data across different clients, and still faces privacy breaches. Therefore, in order to address the above shortcomings, this study proposes an efficient privacy preserving federated graph neural network method (PPFedGNN) for social network analysis, thereby achieving dual guarantees of model performance and privacy security. In order to obtain global user relevance while protecting privacy, a secure coding-based social subgraph aggregation method (SecureSA) was designed. This method improves model performance and algorithm efficiency through secure encoding and aggregation of different client adjacency relationships. Furthermore, in order to reduce the impact of inconsistent node embeddings among different client users, a local differential privacy(LDP) mechanism protected social node augmentation method (SecureNA) was designed. This method improves model performance while protecting security by adding noise perturbations to some important weights and integrating node embedding features from different clients. Through theoretical analysis and experimental verification, it has been found that on social datasets such as Facebook, Blogcatalog, and Flickr, PPFedGNN achieves accuracy of over 0.89, 0.73, and 0.47 for both random and fixed partitioning of social user nodes, respectively. Through ablation experiments, the effectiveness of the SecureSA and SecureNA methods was further demonstrated. In addition, we also conducted a theoretical analysis of the security techniques used throughout the training process to demonstrate their safety and efficiency.


# Framework
![The Framework of PPFedGNN](./framework.png)

# DataSet
The experimental dataset includes BlogCatalog[1], Flikr[1], Facebook[2] datasets, you can download it yourself on the Internet.

# Experimental environment
+ torch == 2.2.2
+ pandas == 1.22.0
+ networkx == 3.1
+ matplotlib == 3.7.1
+ numpy == 1.22.0

# Reference
- [1] L. Tang and H. Liu, “Relational learning via latent social dimensions,” in Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining, 2009, pp. 817–826.
- [2] B. Viswanath, A. Mislove, M. Cha, and K. P. Gummadi, “On the evolution of user interaction in facebook,” in Proceedings of the 2nd ACM workshop on Online social networks, 2009, pp. 37–42.
