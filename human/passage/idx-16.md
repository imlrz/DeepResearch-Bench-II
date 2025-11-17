# Machine Learning-Based Materials Inverse Design: Methods and Databases

## 1. Introduction to Materials Inverse Design

### 1.1 Concept and Significance

Materials inverse design represents a paradigm shift in materials science, fundamentally reversing the traditional forward design approach [(13)](https://www.researchgate.net/journal/Computers-Materials-Continua-1546-2226/publication/388024287_Machine_Learning-Based_Methods_for_Materials_Inverse_Design_A_Review/links/67c168ea8311ce680c76fc2d/Machine-Learning-Based-Methods-for-Materials-Inverse-Design-A-Review.pdf). Unlike forward design, which starts from material composition and structure (ACS) to predict properties through experiments and simulations (ACS→P(ACS)), inverse design begins with desired material properties and works backward to determine the composition and structure (P(ACS)→ACS) [(13)](https://www.researchgate.net/journal/Computers-Materials-Continua-1546-2226/publication/388024287_Machine_Learning-Based_Methods_for_Materials_Inverse_Design_A_Review/links/67c168ea8311ce680c76fc2d/Machine-Learning-Based-Methods-for-Materials-Inverse-Design-A-Review.pdf). This approach addresses the critical challenge of efficiently navigating the vast materials space, which is computationally intractable through conventional exploration methods [(5)](https://pubmed.ncbi.nlm.nih.gov/30049875/).

The significance of materials inverse design extends beyond academic research into practical applications across multiple industries. The discovery of new materials can bring enormous societal and technological progress, particularly in areas such as energy storage, catalysis, carbon capture, and semiconductor technology [(5)](https://pubmed.ncbi.nlm.nih.gov/30049875/). Recent advances in artificial intelligence, particularly in machine learning, have enabled the effective characterization of implicit associations between material properties and structures, opening up an efficient paradigm for functional materials design [(3)](https://arxiv.org/pdf/2411.09429).

### 1.2 Development Trends up to Early 2024

The field of machine learning-based materials inverse design has witnessed rapid evolution, with particularly significant advances from 2020 to early 2024. Recent developments include the emergence of sophisticated generative models such as MatterGen, which demonstrates the capability to generate stable, diverse inorganic materials across the periodic table with more than twice the success rate of prior models [(2)](https://arxiv.org/pdf/2312.03687). The introduction of diffusion-based generative processes represents a major breakthrough, producing crystalline structures by gradually refining atom types, coordinates, and periodic lattices [(2)](https://arxiv.org/pdf/2312.03687).

The integration of large language models (LLMs) into materials inverse design has emerged as a transformative trend . Recent works demonstrate how LLMs can be fine-tuned to capture effective foundational distributions of metastable crystals within textual representations, subsequently converted to graph representations for further refinement . The OMat24 dataset, released in 2024, represents a landmark achievement with over 110 million density functional theory (DFT) calculations, providing unprecedented structural and compositional diversity for training advanced machine learning models [(40)](https://hub.baai.ac.cn/view/40795).

Notable milestones in 2023-2024 include the development of interpretable generative inverse design methods, such as the random-forest-based RIGID approach, which offers single-shot inverse design capabilities for metamaterials with on-demand functional behaviors [(18)](https://par.nsf.gov/biblio/10615002-generative-inverse-design-metamaterials-functional-responses-interpretable-learning). The CASTING framework introduces continuous action space tree search for materials discovery, employing decision tree-based Monte Carlo Tree Search (MCTS) algorithms with continuous space adaptation . These advances collectively demonstrate the field's progression toward more efficient, interpretable, and scalable inverse design methodologies.

## 2. Main Methods for Machine Learning-Based Materials Inverse Design

### 2.1 Exploration-Based Methods

#### 2.1.1 Fundamental Principles

Exploration-based methods in materials inverse design rely on systematic or semi-random search strategies to discover novel materials within the vast materials space [(8)](https://arxiv.org/pdf/2502.02905). These approaches operate under the principle that comprehensive exploration, guided by statistical principles and computational efficiency, can identify materials with desired properties more effectively than traditional trial-and-error approaches. The fundamental premise is that by leveraging machine learning predictions to guide exploration, researchers can significantly reduce the number of required DFT calculations and experimental trials [(8)](https://arxiv.org/pdf/2502.02905).

Random search techniques form the foundation of exploration-based approaches, often implemented through cost-based algorithms that perform random searches within specified compositional spaces . For example, researchers have employed random search in complex compositions like pyrope (Mg₃Al₂(SiO₄)₃), using unit cells containing 4 formula units, 24 and 48 randomly chosen symmetry operations, and random minsep matrices of 2-3 Å . These methods benefit from the availability of high-throughput computational screening platforms and automated experimental systems that can process large numbers of candidate materials efficiently.

The efficiency of exploration-based methods depends critically on the balance between random sampling and intelligent guidance. While purely random search can be computationally expensive, the integration of machine learning predictions allows for prioritized sampling of regions in materials space that are statistically more likely to contain materials with target properties [(8)](https://arxiv.org/pdf/2502.02905). This approach has proven particularly effective when combined with active learning strategies, where the model adapts its sampling strategy based on previously evaluated candidates.

#### 2.1.2 Key Algorithms

Genetic Algorithms (GAs) represent one of the most widely applied exploration-based methods in materials inverse design . These algorithms mimic natural selection processes by encoding candidate materials and employing crossover, mutation, and selection operations to evolve populations over successive generations . GAs are particularly effective in discrete optimization problems where materials are represented as sets of structural parameters encoded in genetic representations. Their ability to efficiently traverse high-dimensional landscapes makes them well-suited for tuning material properties as continuous variables .

Machine Learning-accelerated Genetic Algorithms (MLaGA) integrate on-the-fly Gaussian processes with traditional GAs, achieving remarkable computational efficiency improvements. This approach reduces the number of DFT calculations by up to 50-fold compared to traditional GA approaches and has been successfully applied to identify optimal chemical ordering within binary alloy nanoparticles . The integration of machine learning surrogates enables rapid evaluation of candidate materials, allowing the genetic algorithm to explore larger search spaces more effectively.

Particle Swarm Optimization (PSO) is another prominent exploration-based algorithm inspired by social behavior of birds or fish . In PSO, candidate solutions, represented as particles, iteratively adjust their positions in the search space based on both their own experiences and those of their neighbors. This cooperative approach allows PSO to explore complex design spaces efficiently and converge to global optima . PSO has been widely applied in materials design for thermal and electronic properties, as well as optimizing complex structures where traditional methods might falter. For instance, PSO is implemented in the CALYPSO package as a crystal structure predictor, efficiently optimizing free energy surfaces and generating crystal structures based solely on chemical composition information .

Monte Carlo Tree Search (MCTS) incorporates elements of randomness and systematic search by exploring decision trees to find optimal solutions . The tree-based search strategy prioritizes promising solution pathways, making it particularly useful in combinatorial problems . MCTS has shown promise in identifying optimal configurations for complex material systems by systematically sampling from possible coordination states and prioritizing promising branches. This approach facilitates efficient exploration of combinatorial search spaces and has been applied to identify stable structures, such as silver impurity segregation in copper, by searching only 1% of all possible configurations .

The Continuous Action Space Tree search for INverse desiGn (CASTING) framework represents a recent advancement in exploration-based methods . CASTING employs a decision tree-based MCTS algorithm with continuous space adaptation through modified policies and sampling . This framework demonstrates scalability and accuracy across diverse materials systems, including metals, covalent systems, and complex oxides, with applications in discovering superhard carbon phases and optimizing multi-objective properties .

#### 2.1.3 Advantages and Disadvantages

The primary advantage of exploration-based methods lies in their ability to discover materials with unexpected properties or structures that might be missed by model-based approaches. These methods excel at exploring large, complex search spaces and can identify materials outside the training data distribution [(8)](https://arxiv.org/pdf/2502.02905). Genetic algorithms, in particular, have demonstrated success in handling discrete variables and complex constraint satisfaction problems common in materials design .

Computational efficiency represents another significant advantage, especially when exploration-based methods are enhanced with machine learning accelerators. MLaGA reduces DFT calculations by up to 50-fold compared to traditional approaches . The CASTING framework achieves efficient exploration through continuous action space adaptation, demonstrating scalability across diverse materials systems including metals, covalent systems, and complex oxides .

However, exploration-based methods face several notable limitations. Computational cost remains a significant challenge, as iterative evaluation of candidate solutions often requires computation of many configurations, which can be prohibitive for large datasets . Premature convergence is another common issue, where algorithms become trapped in local optima, missing other optimized solutions elsewhere in the search space .

Generalization poses a particular challenge for exploration-based inverse design, as many algorithms are applied to systems with predefined chemical compositions . The performance of these methods is highly sensitive to parameter choices, including population size, mutation rate, and selection pressure. The absence of standardized optimization techniques makes parameter tuning a non-trivial task, often requiring extensive trial and error .

### 2.2 Model-Based Methods

#### 2.2.1 Fundamental Principles

Model-based methods in materials inverse design leverage established machine learning models to predict material properties and perform inverse mapping from desired properties to material structures [(3)](https://arxiv.org/pdf/2411.09429). These approaches are grounded in the premise that deep learning methods can effectively characterize the implicit associations between material properties and structures, enabling efficient navigation of the vast and nonlinear chemical spaces in materials . The fundamental principle involves constructing computational models that can learn complex, high-dimensional relationships between material descriptors and target properties, then using these models to generate materials with specific property requirements.

Generative models represent the state-of-the-art in model-based inverse design, with their essence being the approximation of probability distributions from training datasets . These models excel in conditional generation and enable the design of materials with targeted, domain-specific properties . The future of inverse design lies in generative models that offer ultra-high flexibility and efficiency for navigating complex materials design spaces .

The development of model-based methods has been significantly advanced by the availability of large materials databases that fuel applications in atomistic prediction [(7)](https://arxiv.org/pdf/2110.14820). These databases provide the foundational data for training sophisticated machine learning models capable of capturing intricate structure-property relationships. Recent advances in image and spectral data have largely leveraged synthetic data enabled by high-quality forward models as well as generative unsupervised deep learning methods [(7)](https://arxiv.org/pdf/2110.14820).

#### 2.2.2 Key Algorithms

Variational Autoencoders (VAEs) provide control over data generation via latent variables . An autoencoder model includes encoding and decoding networks, where the encoder maps molecules to vectors in lower-dimensional latent spaces, and the decoder maps latent vectors back to original representations . VAEs achieve better generalizability by constraining encoding networks to generate latent vectors following probability distributions, typically Gaussian distributions due to their accessible numerical and theoretical properties .

The latent space represents arguably the most interesting component of VAEs for materials design. Molecules are represented as continuous and differentiable vectors residing on probabilistic manifolds . Latent spaces encode geometric relationships, allowing researchers to sample nearby points to decode similar molecules, with increasing distances corresponding to increasingly dissimilar molecules . This property enables direct gradient-based optimization of properties, as latent spaces are continuous vector spaces .

Generative Adversarial Networks (GANs) employ adversarial training frameworks where generators compete against discriminative models . In this framework, generators attempt to generate synthetic data by sampling noise spaces, while discriminators try to distinguish between synthetic and real data . Both models train alternately, with the generator's goal being to learn to structure noise toward producing data that discriminators cannot classify better than chance .

To bias generation processes with GANs and recurrent neural networks (RNNs), gradients are needed to guide network optimization toward desired properties . These properties can be modeled via neural networks and backpropagated to generators. For incorporating properties from chemoinformatic tools, simulations, or experimental measures, gradient estimators that can backpropagate to generators must be created .

Diffusion models represent a recent breakthrough in generative modeling for materials design. MatterGen introduces a novel diffusion process tailored for crystalline materials, generating samples by learning score networks to reverse fixed corruption processes [(2)](https://arxiv.org/pdf/2312.03687). The forward diffusion process independently corrupts atom types (A), coordinates (X), and lattice (L) to approach physically motivated distributions of random materials [(2)](https://arxiv.org/pdf/2312.03687). An equivariant score network is pre-trained on large datasets of stable material structures to jointly denoise atom types, coordinates, and lattice [(2)](https://arxiv.org/pdf/2312.03687).

Reinforcement Learning (RL) combines concepts from machine learning and optimal control to solve sequential decision-making problems . RL models contain agents that interact with environments, taking actions based on observations and receiving rewards that reflect performance . The goal is to maximize cumulative rewards over time by learning optimal policies . In materials design, RL can be applied to explore complex, high-dimensional design spaces by learning policies that maximize reward functions, such as achieving specific property thresholds .

Graph Neural Networks (GNNs) have emerged as powerful tools for materials property prediction and inverse design. Crystal Graph Convolutional Neural Networks (CGCNN) were among the earliest tools to use graph neural networks for predicting material properties . Compared to DFT calculations, CGCNN achieves comparable or slightly better accuracy in predicting properties such as formation energy, band gap, Fermi level, bulk modulus, shear modulus, and Poisson's ratio, with significantly faster computational speed .

The Atomistic Line Graph Neural Network (ALIGNN) introduces an innovative GNN architecture that alternates message passing between atomic bond graphs and line graphs (capturing bond angles) to explicitly incorporate critical geometric information often overlooked by traditional GNN models . Compared to models solely based on interatomic distances, ALIGNN significantly improves the accuracy of material property predictions .

The Equivariant Crystal Network (ECN) incorporates spatial group symmetries into graph neural networks . ECN emphasizes that symmetry invariance is a necessary condition for practical applications. Experimental results show improved performance compared to previous models, supporting the claim that symmetry provides beneficial inductive bias .

#### 2.2.3 Advantages and Disadvantages

Model-based methods offer several significant advantages for materials inverse design. Generative models provide ultra-high flexibility and efficiency for navigating complex materials design spaces . These methods excel in conditional generation, enabling the design of materials with targeted, domain-specific properties . The ability to learn complex, high-dimensional relationships from large datasets allows model-based approaches to capture intricate structure-property relationships that might be difficult to encode explicitly.

Recent advances in model-based methods have demonstrated remarkable improvements in generating stable materials. MatterGen produces structures that are more than twice as likely to be novel and stable, and more than 15 times closer to local energy minima compared to prior generative models [(2)](https://arxiv.org/pdf/2312.03687). After fine-tuning, MatterGen successfully generates stable, novel materials with desired chemistry, symmetry, as well as mechanical, electronic, and magnetic properties [(2)](https://arxiv.org/pdf/2312.03687).

The integration of advanced architectures such as transformers and large language models has opened new possibilities for materials inverse design. FlowLLM integrates LLMs with Riemannian flow matching to enable design of novel crystalline materials, surpassing existing state-of-the-art methods by over threefold in stable material generation rates .

However, model-based methods face several significant challenges. Data requirements represent a primary limitation, as these methods typically require large amounts of high-quality training data. The availability and quality of training data can significantly impact model performance and generalization capabilities. Model interpretability poses another challenge, as many advanced machine learning architectures, particularly deep neural networks, are often considered "black boxes" that provide limited insight into the underlying physical mechanisms governing structure-property relationships.

Computational demands can be substantial, especially for training and deploying complex generative models. The computational cost of training large models such as transformers or sophisticated diffusion models may be prohibitive for many research groups. Additionally, the time required for inference, particularly for generating complex materials structures, can be significant.

Generalization to unseen materials or property regimes remains challenging. Models trained on existing materials may struggle to predict properties or generate structures for materials outside the training distribution. This limitation is particularly problematic for discovering truly novel materials with unprecedented properties or structural features.

### 2.3 Optimization-Based Methods

#### 2.3.1 Fundamental Principles

Optimization-based methods in materials inverse design formulate the materials discovery problem as a mathematical optimization challenge, where the goal is to find material compositions and structures that maximize or minimize specific objective functions subject to various constraints [(15)](https://arxiv.org/pdf/2201.11168). These approaches are particularly valuable for cases where conventional parameter sweeps or trial-and-error sampling are impractical [(15)](https://arxiv.org/pdf/2201.11168). The fundamental principle involves transforming materials design into constrained optimization problems that can be solved using advanced mathematical algorithms and machine learning accelerators.

Adaptive inverse design represents a sophisticated optimization approach that involves refining design models by iteratively updating them based on results from forward simulations or experimental data . In each iteration, the model generates candidate materials that progressively approach desired property targets, guided by feedback from prior evaluations . The interaction feedback loop between design models and materials property predictions or evaluations is typically realized through forward calculations or experiments, informed by computational simulations, empirical datasets, or automated experiments .

The efficiency of optimization-based methods depends critically on the balance between exploration and exploitation. Bayesian optimization exemplifies this principle by constructing probabilistic surrogate models, often Gaussian processes, to approximate objective functions . These surrogate models estimate uncertainty in objective functions and balance exploration (sampling new areas of design space) with exploitation (focusing on areas likely to yield optimal results) .

#### 2.3.2 Key Algorithms

Bayesian Optimization (BO) represents a powerful framework for global optimization in scenarios where data is scarce or expensive to acquire . BO constructs probabilistic surrogate models, often Gaussian processes, to approximate objective functions. The surrogate model estimates uncertainty in the objective function and balances exploration with exploitation . Key techniques such as acquisition functions, including Expected Improvement (EI), Probability of Improvement (PI), and Upper Confidence Bound (UCB), are used to guide next experiments by selecting points that maximize improvement potential .

Multi-objective Bayesian optimization methods have been developed to efficiently achieve predefined goals. These methods demonstrate that by focusing on achieving specific goals, Bayesian optimization can efficiently accelerate realistic multi-objective design problems with minimal effort [(24)](https://pubmed.ncbi.nlm.nih.gov/34286234/). Benchmarks show that proposed BO methods dramatically reduce the number of experiments needed to achieve goals relative to baseline methods [(24)](https://pubmed.ncbi.nlm.nih.gov/34286234/).

Reinforcement Learning (RL) in optimization-based methods treats deep neural networks as "agents" that learn interactions between input elemental compositions and corresponding material properties . There are two main ways of adopting machine learning methods for design optimization: building surrogate models of design spaces for final exploration (Bayesian optimization method) and utilizing deep neural networks to learn interactions between input elemental compositions and their corresponding material properties .

Physics-informed Reinforcement Learning (PIRL) combines adjoint-based methods with reinforcement learning to improve sample efficiency by an order of magnitude compared to conventional reinforcement learning and overcome local minimum issues [(27)](https://pubmed.ncbi.nlm.nih.gov/39679239/). PIRL demonstrates high sample efficiency, robustness, and the ability to seamlessly incorporate practical device design constraints, offering promising approaches to highly combinatorial freeform device optimization in various physical domains [(27)](https://pubmed.ncbi.nlm.nih.gov/39679239/).

Deep Reinforcement Learning (DRL) coupled with homogenization provides approaches to solving inverse problems through application of deep reinforcement learning [(28)](https://m.x-mol.net/paper/detail/1865608057092141056). This methodology has been applied to solve complex inverse design problems in materials science, demonstrating the potential of reinforcement learning for materials discovery [(28)](https://m.x-mol.net/paper/detail/1865608057092141056).

Evolutionary algorithms, including Genetic Algorithms and Particle Swarm Optimization, also fall under optimization-based methods. These algorithms are often used for maximum property search in inverse analysis . Machine learning optimization algorithms of Bayesian optimization, genetic algorithms, and particle swarm optimization are employed to perform inverse analysis with maximum property search .

#### 2.3.3 Advantages and Disadvantages

Optimization-based methods offer several significant advantages for materials inverse design. Bayesian optimization excels in scenarios with limited data availability, making it particularly valuable for expensive experimental validation scenarios . The ability to balance exploration and exploitation through acquisition functions enables efficient navigation of complex design spaces while minimizing required evaluations .

Reinforcement learning approaches demonstrate exceptional sample efficiency, with PIRL achieving order-of-magnitude improvements compared to conventional methods [(27)](https://pubmed.ncbi.nlm.nih.gov/39679239/). These methods excel at handling complex, high-dimensional design spaces and can learn optimal policies for materials discovery tasks .

The adaptive nature of optimization-based methods allows for continuous improvement through feedback loops. These approaches can adapt to new information and refine their search strategies based on experimental or computational results . This capability is particularly valuable for real-world applications where iterative refinement based on experimental validation is necessary.

However, optimization-based methods face several notable limitations. Computational complexity can be substantial, particularly for high-dimensional optimization problems with many design variables. The cost of evaluating objective functions, especially when requiring expensive DFT calculations or experimental validation, can limit the practical application of these methods.

Convergence to global optima is not guaranteed, and optimization methods may become trapped in local minima, particularly for complex, multi-modal objective functions common in materials science. The choice of optimization algorithm and its parameters can significantly impact performance, requiring expertise in algorithm selection and tuning.

The black-box nature of many optimization methods can limit scientific understanding of the discovered materials. While these methods may identify materials with desired properties, they often provide limited insight into the underlying physical mechanisms or design principles that govern these properties.

## 3. Key Databases for Materials Inverse Design

### 3.1 Major Comprehensive Databases

#### 3.1.1 Materials Project

The Materials Project represents one of the most comprehensive and widely used databases for materials research and inverse design [(33)](https://juejin.cn/post/7494213644873662514). It is a large open online materials dataset containing crystal structures, energy properties, electronic structures, and thermodynamic properties [(33)](https://juejin.cn/post/7494213644873662514). The database covers diverse material representations including photoelectric, mechanical, physicochemical, stability and reactivity, thermodynamic, and magnetic properties [(33)](https://juejin.cn/post/7494213644873662514).

The Materials Project provides data through the AWS OpenData Program, with data organized in three buckets named materials-project-{raw,parsed,build} . The parsed bucket contains objects generated by parsing VASP output files, forming the basis for builder pipelines that create derived high-level data collections served through the MP API and website . All S3 objects in this bucket are serialized pymatgen or emmet python objects, most stored as gzip-compressed JSON files for each MP ID .

The build bucket contains high-level derived data comprising sources for collections available through the MP API . Collections and pre-built objects are versioned by database release dates, with individual documents grouped into gzip-compressed JSONL files . This organization enables efficient data access and ensures compatibility with various computational workflows and machine learning applications.

The Materials Project's commitment to open data access has made it an invaluable resource for the materials science community. The database has enabled numerous breakthrough discoveries and has served as the foundation for countless machine learning models in materials science. Its comprehensive coverage of material properties and structures provides an essential resource for materials inverse design applications.

#### 3.1.2 AFLOW

AFLOW (Automatic FLOW for Materials Discovery) represents a globally available database containing 3,929,948 material compounds with over 817,429,184 calculated properties, with continuous growth [(78)](https://aflowlib.org/). The database encompasses various material types and provides extensive computational data including formation enthalpies (3,479,057), band structures (366,988), Bader charges (172,488), elastic properties (5,650), thermal properties (5,664), and binary (1,738), ternary (30,289), and quaternary (150,659) systems .

AFLOW provides multiple interfaces for data access and analysis. The MendeLIB search application enables sophisticated database queries, while AFLOW-online offers an interface for symmetry analysis, structure comparison, CCE (Chemical Composition Evolution), POCC (Polynomial Order Crystal Chemistry), and other functionalities . The AFLOW Prototype Encyclopedia contains over 1,100 prototypes, providing structural templates for materials discovery .

The AFLOW-ML application supports machine learning applications for PLMF (Polynomial Local Moments Fingerprint), MDF (Materials Data Facility), and ASC (Atomic Simulation Environment) models . AFLOW-CHULL provides convex hull applications for thermodynamic stability and synthesizability analysis . These tools collectively enable comprehensive materials analysis and discovery workflows.

AFLOW's REST-API and AFLUX provide programmatic access to the database, enabling integration with computational workflows and machine learning applications. The database's comprehensive coverage and extensive computational data make it a valuable resource for materials inverse design applications, particularly for applications requiring detailed electronic structure and thermodynamic information.

#### 3.1.3 Open Materials 2024 (OMat24)

Open Materials 2024 (OMat24) represents a revolutionary advancement in materials database development, containing over 110 million density functional theory (DFT) calculations focused on structural and compositional diversity [(40)](https://hub.baai.ac.cn/view/40795). This dataset is one of the largest open datasets for training DFT surrogate models in materials science [(40)](https://hub.baai.ac.cn/view/40795). The OMat24 dataset comprises over 110 million DFT calculations representing diverse structural and compositional data of inorganic bulk materials [(43)](https://www.themoonlight.io/en/review/open-materials-2024-omat24-inorganic-materials-dataset-and-models).

The creation of OMat24 required substantial computational resources, totaling approximately 400 million core hours [(43)](https://www.themoonlight.io/en/review/open-materials-2024-omat24-inorganic-materials-dataset-and-models). The dataset's significance lies in its extensive coverage of non-equilibrium atomic configurations and its ability to capture a wide range of materials properties [(43)](https://www.themoonlight.io/en/review/open-materials-2024-omat24-inorganic-materials-dataset-and-models). This comprehensive coverage enables training of sophisticated machine learning models capable of predicting properties across diverse materials space.

OMat24 contains a combination of DFT single-point calculations, structural relaxations, and molecular dynamic trajectories over diverse sets of inorganic bulk materials . This multi-faceted approach provides rich information about materials' ground states, excited states, and dynamic behaviors, making it particularly valuable for materials inverse design applications requiring accurate property predictions across various conditions.

The dataset's release in 2024 marks a significant milestone in materials science, providing the community with unprecedented access to high-quality computational data. The sheer scale and diversity of OMat24 enable training of advanced machine learning models that can capture complex structure-property relationships, opening new possibilities for materials inverse design and discovery.

#### 3.1.4 JARVIS

The Joint Automated Repository for Various Integrated Simulations (JARVIS) is an infrastructure designed to automate materials discovery and optimization using classical force-field, density functional theory, machine learning, quantum computation calculations, and experiments [(45)](https://jarvis.nist.gov/). Established in January 2017 with funding from NIST's MGI (Measurement Science for Quantum Systems) and CHIPS (Creating Helpful Incentives to Produce Semiconductors) programs, JARVIS has grown to include over 50 publications, 4,500+ citations, 150,000+ users, 1.5 million+ downloads, 80,000+ materials, and 1 million+ properties .

JARVIS provides several integrated databases including JARVIS-DFT, JARVIS-FF (force fields), JARVIS-QETB (quantum exact tight binding), JARVIS-ChemNLP (chemical natural language processing), JARVIS-Leaderboard, JARVIS-Tools-Notebooks, JARVIS-OPTIMADE, JARVIS-SuperConDB (superconductors), JARVIS-InterfacDB (interfaces), JARVIS-Universal Search, JARVIS-DMFT (dynamical mean-field theory), FigShare curated datasets, and JARVIS-DFT2 .

The JARVIS-DFT database contains 77,096 materials with OptB88vdW electronic bandgaps and formation energies, 18,293 with TBmBJ bandgaps, 25,513 with elastic tensors, 11,383 with topological SOC spillage, 4,801 with infrared intensities, 15,860 with dielectric functions, 812 with 2D exfoliation energies, 17,642 with carrier effective masses, 4,801 with piezoelectric tensors, 23,210 with Seebeck coefficients, and 11,865 with electric field gradients .

JARVIS has developed several web applications including ALIGNN property predictor, ALIGNN Force-field, solar cells design tools, direct air capture applications, scanning tunneling microscopy and scanning transmission electron microscopy simulation tools, heterostructure design applications, catalysis analysis tools, visualization tools, XRD simulation tools, and battery design applications . These applications provide user-friendly interfaces for materials discovery and inverse design workflows.

The JARVIS platform includes several tools including JARVIS-Tools, ALIGNN, JARVIS-Leaderboard, AtomGPT, ChemNLP, AtomVision, AtomQC, InterMat, DefectMat, and TB3Py . These tools enable comprehensive materials analysis and provide the foundation for advanced machine learning applications in materials science.

#### 3.1.5 Open Quantum Materials Database (OQMD)

The Open Quantum Materials Database (OQMD) contains 1,226,781 materials with DFT-calculated thermodynamic and structural properties . OQMD is a high-throughput database comprising nearly 300,000 DFT total energy calculations of compounds from the Inorganic Crystal Structure Database (ICSD) and decorations of commonly occurring crystal structures [(53)](https://www.mendeley.com/catalogue/4291691e-8839-30f5-8ba1-46f442f6550c/).

OQMD enables users to search materials by composition, create phase diagrams, determine ground state compositions (GCLP), and visualize crystal structures [(56)](https://ce.northwestern.edu/magazine/fall-2014/materials-database.html). These capabilities make OQMD particularly valuable for materials discovery applications requiring thermodynamic stability analysis and phase diagram construction.

The database provides several analysis tools including phase diagram creation, ground state composition determination (GCLP), and crystal structure visualization . These tools enable comprehensive materials analysis and support various materials discovery workflows. OQMD's focus on thermodynamic stability makes it particularly valuable for materials inverse design applications requiring stability constraints.

The database's integration with other resources such as the ICSD ensures comprehensive coverage of known inorganic materials. OQMD's RESTful API (OPTIMADE) provides programmatic access to the database, enabling integration with computational workflows and machine learning applications.

### 3.2 Specialized Databases

#### 3.2.1 Inorganic Crystal Structure Database (ICSD)

The Inorganic Crystal Structure Database (ICSD) represents the world's largest database for completely identified inorganic crystal structures [(57)](https://icsd.products.fiz-karlsruhe.de/en/startpage-icsd). Maintained by FIZ Karlsruhe, ICSD provides scientific and industrial communities with comprehensive crystal structure data, with first records dating back to 1913 . Only data that have passed thorough quality checks are included, ensuring high data reliability .

As of recent updates, ICSD contains 318,901 crystal structures . The database adds approximately 12,000 new structures annually, with continuous quality assurance processes that modify, supplement, or remove duplicates from existing content . This continuous update process ensures that even older content remains current and accurate.

ICSD provides all important crystal structure data including unit cells, space groups, complete atomic parameters, site occupation factors, Wyckoff sequences, molecular formulas and weights, ANX formulas, mineral groups, and other information . Approximately 80% of structures are allocated to about 9,000 structure types, enabling searches for substance classes . The database also provides continuous selection and evaluation of theoretical structures that can serve as bases for developing new materials through data mining processes .

ICSD's comprehensive coverage and high-quality data make it an essential resource for materials science research. The database provides keywords describing physical and chemical properties, abstracts for quick content understanding, and powder diffraction data simulation capabilities . These features support various materials analysis and discovery applications, including materials inverse design workflows that require accurate structural information.

#### 3.2.2 Computational 2D Materials Database (C2DB)

The Computational 2D Materials Database (C2DB) contains approximately 4,000 two-dimensional materials distributed across 40+ different crystal structures, providing structural, thermodynamic, elastic, electronic, magnetic, and optical properties [(62)](https://blog.csdn.net/sinat_26809255/article/details/135412608). The database includes diverse material types such as MXY Janus, MXene, TMDC-H, TMDC-T, TMDC-alloy, Xane, and Xene structures [(62)](https://blog.csdn.net/sinat_26809255/article/details/135412608).

C2DB is a highly curated open database organizing computed properties for more than 4,000 atomically thin two-dimensional materials [(63)](https://zendy.io/title/10.1088/2053-1583/ac1059). The database has undergone continuous development since its first release in 2018, with regular additions of new materials and properties [(63)](https://zendy.io/title/10.1088/2053-1583/ac1059). This ongoing development ensures that C2DB remains current with emerging 2D materials research.

The C2DB dataset of approximately 4,000 entries provides reliable data based on DFT calculations, supporting development of novel descriptors for boosting prediction of electronic structure properties of 2D materials . The database's comprehensive coverage of 2D materials properties makes it particularly valuable for applications in electronics, optoelectronics, and energy storage devices.

C2DB's focus on 2D materials provides specialized data for applications requiring knowledge of atomic-layered materials. The database's organization of materials by structure types enables efficient searching and comparison of similar materials. The comprehensive property data supports development of machine learning models for 2D materials discovery and inverse design applications.

#### 3.2.3 Magnetic Materials Database

The Magnetic Materials Database contains 39,822 magnetic materials with magnetic transition temperatures [(36)](https://ai4science.io/2024physics.html). This specialized database provides comprehensive information about magnetic materials and their properties, enabling researchers to explore and analyze magnetic materials for various applications .

Each entry in the database includes material chemical compounds, related structures (space group, crystal structure), and magnetic temperatures (Curie, Néel, and other transitional temperatures) . This detailed information enables comprehensive analysis of magnetic materials and their phase transitions.

The Northeast Materials Database (NEMAD) represents a comprehensive, experiment-based magnetic materials database containing 26,706 magnetic materials [(68)](https://arxiv.org/pdf/2409.15675v1). NEMAD uses large language models to create comprehensive, experiment-based databases of magnetic materials, providing valuable resources for magnetic materials research and development [(68)](https://arxiv.org/pdf/2409.15675v1).

Specialized databases such as the TCP Mag 2 thermodynamic and properties database are designed for permanent magnetic NdFeB-based alloys [(69)](https://thermocalc.com/products/databases/permanent-magnetic-materials/). These databases provide detailed thermodynamic data including Gibbs energy, volume, and liquid viscosity properties [(69)](https://thermocalc.com/products/databases/permanent-magnetic-materials/). Such specialized databases enable targeted research and development of magnetic materials for specific applications.

The Magnetic Materials Database's focus on magnetic properties and transition temperatures makes it particularly valuable for applications in electronics, energy storage, and magnetic devices. The database's organization by magnetic properties enables efficient searching for materials with specific magnetic characteristics, supporting materials inverse design applications targeting magnetic functionality.

#### 3.2.4 Materials Data Facility (MDF)

The Materials Data Facility (MDF) serves as a platform for publishing, discovering, and accessing materials datasets [(71)](https://www.materialsdatafacility.org/). MDF development and operations are supported by NIST, enabling creation of a national data infrastructure for materials science [(71)](https://www.materialsdatafacility.org/). The facility operates two cloud-hosted services: data publication and data discovery, with features promoting open data sharing, self-service data publication and curation, and encouraging data reuse [(72)](https://www.anl.gov/argonne-scientific-publications/pub/131746).

MDF provides hosting for datasets ranging from kilobytes to terabytes in size, making it easy to share and access large materials datasets . When datasets are published, they receive permanent identifiers (such as DOIs) to simplify citation . The platform's infrastructure is built for accessibility, using Globus to enable easy data transfer to various destinations, from laptops to supercomputers .

The MDF currently hosts over 650 datasets comprising more than 80 TB of materials data, with over 100 data sources indexed . This extensive collection provides diverse resources for materials research and development. The platform's data discovery capabilities enable researchers to find relevant datasets through web interfaces or SDKs .

MDF's commitment to open data sharing and FAIR (Findable, Accessible, Interoperable, Reusable) principles makes it a valuable resource for the materials science community. The platform supports various data formats and provides tools for data aggregation, sharing, and analysis. MDF's integration with other resources and its support for machine learning applications make it particularly valuable for materials inverse design workflows.

#### 3.2.5 NOMAD Repository

The Novel Materials Discovery (NOMAD) repository maintains the largest repository for input and output files of all important computational materials science codes . NOMAD is a web application and database that allows central publication of materials science data [(77)](http://nomad-lab.eu/prod/v1/test/docs/archive.html). From its open-access data, NOMAD builds several big-data services that help advance materials science and engineering .

NOMAD enables users to manage and share materials science data in ways that make it useful for individuals, groups, and the community [(76)](https://nomad-lab.eu/nomad-lab/). The platform is free and open source, encouraging community participation and collaboration [(76)](https://nomad-lab.eu/nomad-lab/). NOMAD processes files to extract structured data and rich metadata, providing a unified way to Find, Access, Interoperate with, and Reuse millions of FAIR data from different codes, sources, and workflows .

Currently, NOMAD has processed 19,218,236 uploaded entries representing 4,341,443 materials, with 113.3 TB of uploaded files . The platform processes files from over 60 formats, enabling integration of diverse data sources . NOMAD provides capabilities for incremental dataset creation, customizable electronic lab notebooks (ELNs), and data publication with DOIs .

NOMAD's comprehensive data processing capabilities enable extraction of structured data and metadata from various file formats. The platform provides unified access to materials data through web interfaces and APIs, supporting both human and machine consumption. NOMAD's integration with machine learning tools and its support for data analysis workflows make it valuable for materials inverse design applications.

The platform offers two deployment options: NOMAD and NOMAD Oasis. NOMAD provides an open and free service for managing, sharing, and publishing data, while NOMAD Oasis enables local data management with user-controlled resources and rules . This dual approach provides flexibility for different user needs and deployment scenarios.

## 4. Conclusion and Future Perspectives

The field of machine learning-based materials inverse design has undergone remarkable transformation from 2020 to early 2024, establishing itself as a cornerstone of modern materials science research. The three primary methodological categories—exploration-based, model-based, and optimization-based approaches—each offer distinct advantages and address specific challenges in materials discovery. Exploration-based methods excel at discovering unexpected materials through systematic search strategies, model-based methods leverage sophisticated machine learning architectures to learn complex structure-property relationships, and optimization-based methods provide mathematical frameworks for navigating constrained design spaces.

The development of advanced algorithms within each category demonstrates significant progress. Genetic algorithms and Monte Carlo Tree Search have been enhanced through machine learning integration, achieving up to 50-fold reductions in computational requirements . Generative models such as MatterGen produce materials with more than twice the success rate of prior approaches while being over 15 times closer to local energy minima [(2)](https://arxiv.org/pdf/2312.03687). Reinforcement learning methods like PIRL achieve order-of-magnitude improvements in sample efficiency compared to conventional approaches [(27)](https://pubmed.ncbi.nlm.nih.gov/39679239/).

The emergence of large-scale databases has fundamentally transformed the field's capabilities. The OMat24 dataset's 110 million DFT calculations [(40)](https://hub.baai.ac.cn/view/40795), combined with established resources like the Materials Project [(33)](https://juejin.cn/post/7494213644873662514), AFLOW [(78)](https://aflowlib.org/), and JARVIS [(45)](https://jarvis.nist.gov/), provide unprecedented foundations for training and validating machine learning models. These databases enable exploration of materials space at scales and resolutions previously impossible, while specialized resources like C2DB for 2D materials [(62)](https://blog.csdn.net/sinat_26809255/article/details/135412608) and magnetic materials databases [(36)](https://ai4science.io/2024physics.html) support targeted applications.

Looking toward the future, several trends are likely to shape the field's development. The integration of large language models with materials science represents a promising direction, as demonstrated by FlowLLM's achievement in surpassing state-of-the-art methods by over threefold in stable material generation rates . Continued advances in generative models, particularly diffusion models and transformers tailored for materials applications, will likely enable more sophisticated inverse design capabilities.

The convergence of experimental automation with machine learning guidance offers opportunities for closed-loop materials discovery systems. The A-Lab's integration of computations, text mining, robotic synthesis, recipe optimization, and phase identification  exemplifies this trend. As experimental throughput increases and costs decrease, the ability to rapidly validate machine learning predictions will become increasingly important.

Challenges remain in several areas. Model interpretability continues to be a concern, as many advanced machine learning architectures provide limited insight into underlying physical mechanisms. The generalization of models to materials outside training distributions requires continued research, particularly for discovering truly novel materials with unprecedented properties. Computational efficiency, while improving, remains a barrier for large-scale applications, necessitating continued advances in algorithm optimization and hardware acceleration.

The field's evolution toward more interdisciplinary approaches, combining physics-informed modeling with data-driven methods, promises to yield more robust and interpretable results. The integration of domain knowledge into machine learning architectures, such as symmetry constraints in ECN  and physics-informed loss functions, represents important progress in this direction.

In conclusion, machine learning-based materials inverse design has evolved from an emerging methodology to a mature field with demonstrated capabilities in discovering novel materials across diverse applications. The continued development of advanced algorithms, large-scale databases, and integrated experimental-computational platforms positions the field for continued breakthroughs in materials discovery and design. As these capabilities mature, materials inverse design is poised to play an increasingly central role in addressing global challenges in energy, electronics, sustainability, and advanced manufacturing.

**References&#x20;**

\[1] What can machine learning help with microstructure-informed materials modeling and design?[ https://arxiv.org/pdf/2405.18396](https://arxiv.org/pdf/2405.18396)

\[2] MatterGen: a generative model for inorganic materials design[ https://arxiv.org/pdf/2312.03687](https://arxiv.org/pdf/2312.03687)

\[3] ai-driven inverse design of materials: past, present and future[ https://arxiv.org/pdf/2411.09429](https://arxiv.org/pdf/2411.09429)

\[4] Methods and applications of machine learning in computational design of optoelectronic semiconductors[ https://dds.sciengine.com/cfs/files/pdfs/view/2095-8226/2EEE2F91EDB8496B8E869B4E7A85FE67.pdf](https://dds.sciengine.com/cfs/files/pdfs/view/2095-8226/2EEE2F91EDB8496B8E869B4E7A85FE67.pdf)

\[5] Inverse molecular design using machine learning: Generative models for matter engineering[ https://pubmed.ncbi.nlm.nih.gov/30049875/](https://pubmed.ncbi.nlm.nih.gov/30049875/)

\[6] Artificial intelligence and machine learning in design of mechanical materials[ https://typeset.io/pdf/artificial-intelligence-and-machine-learning-in-design-of-4c7zvoengq.pdf](https://typeset.io/pdf/artificial-intelligence-and-machine-learning-in-design-of-4c7zvoengq.pdf)

\[7] Recent Advances and Applications of Deep Learning Methods in Materials Science[ https://arxiv.org/pdf/2110.14820](https://arxiv.org/pdf/2110.14820)

\[8] AI-driven materials design: a mini-review[ https://arxiv.org/pdf/2502.02905](https://arxiv.org/pdf/2502.02905)

\[9] Gradient-Based Optimization of Core-Shell Particles with Discrete Materials for Directional Scattering[ https://arxiv.org/pdf/2502.13338](https://arxiv.org/pdf/2502.13338)

\[10] Reinforcement-learning-based Algorithms for Optimization Problems and Applications to Inverse Problems[ https://arxiv.org/pdf/2310.06711](https://arxiv.org/pdf/2310.06711)

\[11] Dakota, A Multilevel Parallel Object-Oriented Framework for Design Optimization, Parameter Estimation, Uncertainty Quantification, and Sensitivity Analysis: Version 6.15 Theory Manual[ https://salix.enialis.net/x86\_64/extra-15.0/source/academic/dakota/Theory-6.15.0.pdf](https://salix.enialis.net/x86_64/extra-15.0/source/academic/dakota/Theory-6.15.0.pdf)

\[12] benchmarking inverse optimization algorithms for materials design[ https://pubs.aip.org/aip/apm/article-pdf/doi/10.1063/5.0177266/19462859/021107\_1\_5.0177266.am.pdf](https://pubs.aip.org/aip/apm/article-pdf/doi/10.1063/5.0177266/19462859/021107_1_5.0177266.am.pdf)

\[13] machine learning-based methods for materials inverse design: a review[ https://www.researchgate.net/journal/Computers-Materials-Continua-1546-2226/publication/388024287\_Machine\_Learning-Based\_Methods\_for\_Materials\_Inverse\_Design\_A\_Review/links/67c168ea8311ce680c76fc2d/Machine-Learning-Based-Methods-for-Materials-Inverse-Design-A-Review.pdf](https://www.researchgate.net/journal/Computers-Materials-Continua-1546-2226/publication/388024287_Machine_Learning-Based_Methods_for_Materials_Inverse_Design_A_Review/links/67c168ea8311ce680c76fc2d/Machine-Learning-Based-Methods-for-Materials-Inverse-Design-A-Review.pdf)

\[14] What can machine learning help with microstructure-informed materials modeling and design?[ https://arxiv.org/pdf/2405.18396](https://arxiv.org/pdf/2405.18396)

\[15] Machine learning-assisted design of material properties[ https://arxiv.org/pdf/2201.11168](https://arxiv.org/pdf/2201.11168)

\[16] dZiner: Rational Inverse Design of Materials with AI Agents[ https://arxiv.org/pdf/2410.03963](https://arxiv.org/pdf/2410.03963)

\[17] Materials Informatics: An Algorithmic Design Rule[ https://arxiv.org/pdf/2305.03797](https://arxiv.org/pdf/2305.03797)

\[18] Generative Inverse Design of Metamaterials with Functional Responses by Interpretable Learning[ https://par.nsf.gov/biblio/10615002-generative-inverse-design-metamaterials-functional-responses-interpretable-learning](https://par.nsf.gov/biblio/10615002-generative-inverse-design-metamaterials-functional-responses-interpretable-learning)

\[19] 【AI for Science|Adv. Mater.】基于神经算子表征与反向设计随机力学超材料 - 腾讯云开发者社区-腾讯云[ https://cloud.tencent.com.cn/developer/news/2922468](https://cloud.tencent.com.cn/developer/news/2922468)

\[20] 三大深度学习生成模型:VAE、GAN及其变种\_vae gan transformer-CSDN博客[ https://blog.csdn.net/heyc861221/article/details/80130968](https://blog.csdn.net/heyc861221/article/details/80130968)

\[21] Recent advances in the inverse design of silicon photonic devices and related platforms using deep generative models - PubMed[ https://pubmed.ncbi.nlm.nih.gov/40567675/](https://pubmed.ncbi.nlm.nih.gov/40567675/)

\[22] Deep generative model for the inverse design of Van der Waals heterostructures - PubMed[ https://pubmed.ncbi.nlm.nih.gov/40594442/](https://pubmed.ncbi.nlm.nih.gov/40594442/)

\[23] 贝叶斯优化与遗传算法:共同点与区别-CSDN博客[ https://blog.csdn.net/universsky2015/article/details/135810474](https://blog.csdn.net/universsky2015/article/details/135810474)

\[24] Bayesian optimization for goal-oriented multi-objective inverse material design - PubMed[ https://pubmed.ncbi.nlm.nih.gov/34286234/](https://pubmed.ncbi.nlm.nih.gov/34286234/)

\[25] Optimization and Supervised Machine Learning Methods for Inverse Design of Cellular Mechanical Metamaterials[ https://vtechworks.lib.vt.edu/items/2dd3fa05-34e5-4cc1-be9c-f9ea87f6d676](https://vtechworks.lib.vt.edu/items/2dd3fa05-34e5-4cc1-be9c-f9ea87f6d676)

\[26] 逆强化学习论文笔记 (一)-CSDN博客[ https://blog.csdn.net/lan\_12138/article/details/118497160](https://blog.csdn.net/lan_12138/article/details/118497160)

\[27] Sample-efficient inverse design of freeform nanophotonic devices with physics-informed reinforcement learning - PubMed[ https://pubmed.ncbi.nlm.nih.gov/39679239/](https://pubmed.ncbi.nlm.nih.gov/39679239/)

\[28] Inverse material design using deep reinforcement learning and homogenization[ https://m.x-mol.net/paper/detail/1865608057092141056](https://m.x-mol.net/paper/detail/1865608057092141056)

\[29] Deep reinforcement learning for inverse inorganic materials design[ https://m.x-mol.net/paper/detail/1869830855485489152](https://m.x-mol.net/paper/detail/1869830855485489152)

\[30] PolyRL: Reinforcement Learning-Guided Polymer Generation for Multi-Objective Polymer Discovery[ https://www.cambridge.org/engage/chemrxiv/article-details/683d8b6c1a8f9bdab53f08a8](https://www.cambridge.org/engage/chemrxiv/article-details/683d8b6c1a8f9bdab53f08a8)

\[31] AWS OpenData[ https://docs.materialsproject.org/downloading-data/aws-opendata](https://docs.materialsproject.org/downloading-data/aws-opendata)

\[32] Aflow - Automatic FLOW for Materials Discovery[ https://aflowlib.org/](https://aflowlib.org/)

\[33] 从数据集到开源模型，覆盖无机材料设计/晶体结构预测/材料属性记录等HyperAI超神经为大家整理了当下热门的材料数据集以 - 掘金[ https://juejin.cn/post/7494213644873662514](https://juejin.cn/post/7494213644873662514)

\[34] New materials discovery using (pdf)[ https://www.energy.gov/sites/default/files/2024-06/New%20materials%20discovery%20using%20simulation%2C%20machine%20learning%2C%20and%20automated%20laboratories%20by%20Anubhav%20Jain.pdf](https://www.energy.gov/sites/default/files/2024-06/New%20materials%20discovery%20using%20simulation%2C%20machine%20learning%2C%20and%20automated%20laboratories%20by%20Anubhav%20Jain.pdf)

\[35] 常用材料五行数据库超全汇总\_物性数据库软件-CSDN博客[ https://blog.csdn.net/sinat\_26809255/article/details/135412608](https://blog.csdn.net/sinat_26809255/article/details/135412608)

\[36] AI 4 Science[ https://ai4science.io/2024physics.html](https://ai4science.io/2024physics.html)

\[37] Center for Hierarchical Materials Design[ https://chimad.northwestern.edu/research/databases.html](https://chimad.northwestern.edu/research/databases.html)

\[38] MatAi-Materials powered by AI[ https://www.mat.ai/news/detail/0b6737a6-031d-4ba2-9e17-97e49ddcd718](https://www.mat.ai/news/detail/0b6737a6-031d-4ba2-9e17-97e49ddcd718)

\[39] The MAterials Simulation Toolkit for Machine Learning (MAST-ML): Automating Development and Evaluation of Machine Learning Models for Materials Property Prediction(pdf)[ https://pdfs.semanticscholar.org/06ad/24a41c8d99984b6112b70e02e673609ff79d.pdf](https://pdfs.semanticscholar.org/06ad/24a41c8d99984b6112b70e02e673609ff79d.pdf)

\[40] 几乎覆盖元素周期表!Meta 发布开源 OMat24 数据集，含 1.1 亿 DFT 计算结果 - 智源社区[ https://hub.baai.ac.cn/view/40795](https://hub.baai.ac.cn/view/40795)

\[41] OMat24: Accelerating AI-Driven Inorganic Material Discovery[ https://www.onlinetools.directory/omat24-ai-inorganic-material-discovery-dataset/](https://www.onlinetools.directory/omat24-ai-inorganic-material-discovery-dataset/)

\[42] Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models(pdf)[ https://arxiv.org/pdf/2410.12771v1](https://arxiv.org/pdf/2410.12771v1)

\[43] \[Literature Review] Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models[ https://www.themoonlight.io/en/review/open-materials-2024-omat24-inorganic-materials-dataset-and-models](https://www.themoonlight.io/en/review/open-materials-2024-omat24-inorganic-materials-dataset-and-models)

\[44] Meta AI Unveils OMAT24: A Groundbreaking Dataset for Inorganic Materials[ https://toolhunt.io/meta-ai-unveils-omat24-a-groundbreaking-dataset-for-inorganic-materials/](https://toolhunt.io/meta-ai-unveils-omat24-a-groundbreaking-dataset-for-inorganic-materials/)

\[45] NIST-JARVIS[ https://jarvis.nist.gov/](https://jarvis.nist.gov/)

\[46] ML[ https://jarvis-materials-design.github.io/dbdocs/jarvisml/](https://jarvis-materials-design.github.io/dbdocs/jarvisml/)

\[47] Machine learning with force-field inspired descriptors for materials: fast screening and mapping energy landscape(pdf)[ https://arxiv.org/pdf/1805.07325v2](https://arxiv.org/pdf/1805.07325v2)

\[48] Accelerated Discovery of Efficient Solar-cell Materials using Quantum and Machine-learning Methods(pdf)[ https://arxiv.org/pdf/1903.06651v2](https://arxiv.org/pdf/1903.06651v2)

\[49] 材料科学人工智能-AI for science\_jarvis数据集-CSDN博客[ https://blog.csdn.net/m0\_70087562/article/details/144054171](https://blog.csdn.net/m0_70087562/article/details/144054171)

\[50] OPTIMADE provider "Joint Automated Repository for Various Integrated Simulations (JARVIS)" (id: jarvis)[ https://www.optimade.org/providers-dashboard/providers/jarvis.html](https://www.optimade.org/providers-dashboard/providers/jarvis.html)

\[51] The Open Quantum Materials Database (OQMD)\_数据集[ https://www.selectdataset.com/dataset/207dbd3b55291da7e6ad33c99851c6d0](https://www.selectdataset.com/dataset/207dbd3b55291da7e6ad33c99851c6d0)

\[52] OPTIMADE provider "The Open Quantum Materials Database (OQMD)" (id: oqmd)[ https://www.optimade.org/providers-dashboard/providers/oqmd.html](https://www.optimade.org/providers-dashboard/providers/oqmd.html)

\[53] The Open Quantum Materials Database (OQMD): Assessing the accuracy of DFT formation energies[ https://www.mendeley.com/catalogue/4291691e-8839-30f5-8ba1-46f442f6550c/](https://www.mendeley.com/catalogue/4291691e-8839-30f5-8ba1-46f442f6550c/)

\[54] Reflections on one million compounds in the open quantum materials database (OQMD)[ https://m.x-mol.net/paper/detail/1616276817061167104](https://m.x-mol.net/paper/detail/1616276817061167104)

\[55] Reflections on one million compounds in the open quantum materials database (OQMD)[ https://ouci.dntb.gov.ua/en/works/4OaWyJq9/](https://ouci.dntb.gov.ua/en/works/4OaWyJq9/)

\[56] World's Largest Materials Database Now Open[ https://ce.northwestern.edu/magazine/fall-2014/materials-database.html](https://ce.northwestern.edu/magazine/fall-2014/materials-database.html)

\[57] Home | ICSD[ https://icsd.products.fiz-karlsruhe.de/en/startpage-icsd](https://icsd.products.fiz-karlsruhe.de/en/startpage-icsd)

\[58] ICSD News | ICSD[ https://icsd.products.fiz-karlsruhe.de/aktuelles/icsd-news?page=0#main-content](https://icsd.products.fiz-karlsruhe.de/aktuelles/icsd-news?page=0#main-content)

\[59] 有机晶体数据库\_福利干货:晶体学数据库大盘点-CSDN博客[ https://blog.csdn.net/weixin\_32154109/article/details/113719802](https://blog.csdn.net/weixin_32154109/article/details/113719802)

\[60] About ICSD[ https://icsd.products.fiz-karlsruhe.de/en/high-contrast/enable?destination=%2Fen%2Fabout%2Fabout-icsd](https://icsd.products.fiz-karlsruhe.de/en/high-contrast/enable?destination=%2Fen%2Fabout%2Fabout-icsd)

\[61] Inorganic Crystal Structure Database[ https://www.re3data.org/repository/r3d100010085](https://www.re3data.org/repository/r3d100010085)

\[62] 常用材料五行数据库超全汇总\_物性数据库软件-CSDN博客[ https://blog.csdn.net/sinat\_26809255/article/details/135412608](https://blog.csdn.net/sinat_26809255/article/details/135412608)

\[63] Recent Progress of the Computational 2D Materials Database (C2DB)[ https://zendy.io/title/10.1088/2053-1583/ac1059](https://zendy.io/title/10.1088/2053-1583/ac1059)

\[64] Anisotropic properties of monolayer 2D materials: an overview from the C2DB database(pdf)[ https://www.researchgate.net/profile/Luca-Vannucci/publication/342801012\_Anisotropic\_properties\_of\_monolayer\_2D\_materials\_an\_overview\_from\_the\_C2DB\_database/links/5f27d7f692851cd302d58495/Anisotropic-properties-of-monolayer-2D-materials-an-overview-from-the-C2DB-database.pdf](https://www.researchgate.net/profile/Luca-Vannucci/publication/342801012_Anisotropic_properties_of_monolayer_2D_materials_an_overview_from_the_C2DB_database/links/5f27d7f692851cd302d58495/Anisotropic-properties-of-monolayer-2D-materials-an-overview-from-the-C2DB-database.pdf)

\[65] The Computational 2D Materials Database: high-throughput modeling and discovery of atomically thin crystals[ https://scite.ai/reports/the-computational-2d-materials-database-XxbG2Gw](https://scite.ai/reports/the-computational-2d-materials-database-XxbG2Gw)

\[66] 国家基础学科公共科学数据中心[ https://www.nbsdc.cn/general/dataDetail?id=646f076887c4325ccc97263b\&type=1](https://www.nbsdc.cn/general/dataDetail?id=646f076887c4325ccc97263b\&type=1)

\[67] MagWeb Products that you can use[ https://www.magweb.us/license-bh-mag-databse/](https://www.magweb.us/license-bh-mag-databse/)

\[68] Northeast Materials Database (NEMAD): Enabling Discovery of High Transition Temperature Magnetic Compounds(pdf)[ https://arxiv.org/pdf/2409.15675v1](https://arxiv.org/pdf/2409.15675v1)

\[69] Permanent Magnetic Materials Database[ https://thermocalc.com/products/databases/permanent-magnetic-materials/](https://thermocalc.com/products/databases/permanent-magnetic-materials/)

\[70] Contribute Data[ https://www.magweb.us/contribute-data/](https://www.magweb.us/contribute-data/)

\[71] Materials Data Facility (MDF) | Publish, Discover & Access Materials Data[ https://www.materialsdatafacility.org/](https://www.materialsdatafacility.org/)

\[72] The Materials Data Facility: Data services to advance materials science research[ https://www.anl.gov/argonne-scientific-publications/pub/131746](https://www.anl.gov/argonne-scientific-publications/pub/131746)

\[73] Materials Data Facility - Data services to advance materials science research[ https://experts.illinois.edu/en/publications/materials-data-facility-data-services-to-advance-materials-scienc](https://experts.illinois.edu/en/publications/materials-data-facility-data-services-to-advance-materials-scienc)

\[74] Manufacturing Demonstration Facility (MDF) at Oak Ridge National Laboratory[ https://www.energy.gov/eere/ammto/manufacturing-demonstration-facility-mdf-oak-ridge-national-laboratory](https://www.energy.gov/eere/ammto/manufacturing-demonstration-facility-mdf-oak-ridge-national-laboratory)

\[75] NOMAD– Manage and Publish Materials Data[ https://nomad-lab.eu/prod/v1/gui/about](https://nomad-lab.eu/prod/v1/gui/about)

\[76] NOMAD[ https://nomad-lab.eu/nomad-lab/](https://nomad-lab.eu/nomad-lab/)

\[77] Materials science data managed and shared¶[ http://nomad-lab.eu/prod/v1/test/docs/archive.html](http://nomad-lab.eu/prod/v1/test/docs/archive.html)

\[78] Aflow - Automatic FLOW for Materials Discovery[ https://aflowlib.org/](https://aflowlib.org/)

\[79] Materials Database Access: AFLOW.org, AFLOW REST-API, AFLUX(pdf)[ https://www.aflowlib.org/aflow-school/past\_schools/20200723/1\_aflow\_school\_database\_aflux.pdf](https://www.aflowlib.org/aflow-school/past_schools/20200723/1_aflow_school_database_aflux.pdf)

\[80] AFLOWLIB\_数据集[ https://www.selectdataset.com/dataset/5133bc99eb2a63e11a9ea42db2016d94](https://www.selectdataset.com/dataset/5133bc99eb2a63e11a9ea42db2016d94)

\[81] A Practical Python API for Querying AFLOWLIB(pdf)[ https://arxiv.org/pdf/1710.00813.pdf](https://arxiv.org/pdf/1710.00813.pdf)

\[82] AI-driven inverse design of materials: Past, present and future - 智源社区论文[ https://hub.baai.ac.cn/paper/171301bf-c2c8-4ca3-b5a5-b3a2ad8e47ad](https://hub.baai.ac.cn/paper/171301bf-c2c8-4ca3-b5a5-b3a2ad8e47ad)

\[83] Inverse-designed 3D sequential metamaterials achieving extreme stiffness[ https://research.tudelft.nl/en/publications/inverse-designed-3d-sequential-metamaterials-achieving-extreme-st](https://research.tudelft.nl/en/publications/inverse-designed-3d-sequential-metamaterials-achieving-extreme-st)

\[84] OMat24: Accelerating AI-Driven Inorganic Material Discovery[ https://www.onlinetools.directory/omat24-ai-inorganic-material-discovery-dataset/](https://www.onlinetools.directory/omat24-ai-inorganic-material-discovery-dataset/)

\[85] Building a comprehensive database for Carbon materials: From structural generation to Machine Learning Interatomic Potential (MLIP) training[ https://acs.digitellinc.com/p/s/building-a-comprehensive-database-for-carbon-materials-from-structural-generation-to-machine-learning-interatomic-potential-mlip-training-595430](https://acs.digitellinc.com/p/s/building-a-comprehensive-database-for-carbon-materials-from-structural-generation-to-machine-learning-interatomic-potential-mlip-training-595430)

> (Note: This document may contain AI-generated content.)