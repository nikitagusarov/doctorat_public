# Presentation of the doctoral studies project

**Key words:** Consumer Choice, Econometrics, Preference Studies, Data Science, Machine Learning, Classification Technics

### PhD Candidate:

**Nikita Gusarov** 

Master student in MIASHS-C2ES
 
### Supervision: 

|   Directeur de thèse | Co-encadrant |
|:---------------------|:------------------|
|**Iragaël Joly** |**Pierre Lemaire** |
| (MCF HDR - GAEL) |(MCF - GSCOP) |
| iragael.joly@grenoble-inp.fr | pierre.lemaire@grenoble-inp.fr |
| Econometrics in Consumption Choice Modeling | Data Sciences in Industrial Engineering|





**Start date:** October 2020
               




# Background

The advances in statistical learning (@hastie2009sl), data analysis (@zielesny2011cf) and data science (@donoho2017ds) of the past decades  have resulted in propagation of *Machine Learning* (ML) techniques. 
Nowadays, it is impossible to imagine a field of science that is not benefiting from the fruits of statistical learning.
The works of @depalma2011tr and @cascetta2009tr on transportation modelling, the publications of @molina2019soc dedicated to sociology problematic, the articles of @coussement2010gam concerning marketing decisions, actuary analysis studies (@denuit2019as1, @denuit2019as3) or even psychology with an example of @baayen2017gam work.

In economics, consumer choices data are mainly studied through classification tools from machine learning  techniques or regression tools like discret choice models from econometric techniques. These two pratices illustrate two  distinct approaches to applying statistical learning. As described by @breiman2001stat and later by @athey2019ml: the *Machine Learning* (ML) focus on the predictive qualities and *Econometrics* attempts to decipher the underlying properties of the data.
Engineering sciences and Computer sciences focuses mainly on ML techniques, whereas in Economics, the scientific community prefers to implement the traditional econometrics techniques to explore hidden patterns (@athey2018iml). 
The focus on explainability of results in advanced ML techniques rarely appearing in economics publications because of their believed lack of interpretability.
Nevertheless, some pluri-disciplinary scientists make attempts to breach this wall between *ML* and *Econometrics*: @athey2019ml,  @mullainathan2017ml, @varian2014bd.
Their advances are mostly focused on the general interdisciplinary question, without entering into the application specific details.

Several studies compare the performances of different econometric and ML models in real world scenarios, although there is still no known work proposing a systematic analysis of performance of all baseline models.
The performance of competing models can be studied in terms of the quality of data adjustments,  the predictive capacity, the quality of the economic and behavioural indicators derived from estimates, as well as in terms of algorithmic efficiency and resource usage.
None of the known to us articles manages all these aspects into their benchmarks, limiting their studies only with several performance criteria. Many of them explore the impacts of different specifications on the same observed choice situation (@fiebig2010gmlm, @mccausland2013pd, @joly2019qcm). But few of them [^1] control their data and model accuracy through experimentation or simulation.

[^1]: only @munizaga2005mlyp appears to use simulated data from our first litterature survey

There exists a particular interest to make the focus on the state of art econometric discrete choice models (@agresti2013cd, @agresti2007cd) as well as their counterparts used in machine learning (@hastie2009sl, @kotsiantis2006tr), alimented with the emerging cross-field studies (@molina2019soc).

As the decision making modelling is directly intertwined to the process of decision making, various theoretical aspects specific to behavioural studies has to be considered.
For example, the economic decision theory derives mostly from the random utility theory (RUM) of @mcfadden2001ec. This decision theory was recently challenged by alternative visions such as random regret minimisation theory (RRM) of @chorus2010rrm, or the relative advantage maximisation theory (RAM) of @leong2015ram, or even quantum decision theory (QDT) of @yukalov2017quantum, which offers a wide range of tools for modelling under uncertainty. 
Recent studies exploring and comparing the context aware models against context-free ones (@belgiawan2019cdm), providing evidence of the advantages of the former.
There exist a multitude of other theoretical elements unexplained by the most traditional models that may be incorporated into the decision making framework. 

These different models address various aspects of the decision making process, under different suppositions.
For example, one of the basic assumptions of the traditional choice theory is transitivity of choice, meaning there exists a strict hierarchy of individual preferences among alternatives. 
Evidently, that is not always the case in real world for some sets of alternatives, and this bias is addressed by quantum decision theory, which manages to bypass this shortcoming and incorporate non-transitivity of choices into the framework.
There exist a multitude of other theoretical elements unexplained by the most traditional models that may be incorporated into the decision making framework, such as loss aversion for example, that could be addressed with random regret minimisation theory.

Finally, there exists a multitude of particular cases in modelling individual choices, that require specific techniques to be addressed.
A family of duration models may be used to model the individual decisions over time (@vitetta2016quantum); network modelling that allows to incorporate spatial and social dependencies for the explored data (@brock2003mcsi); preference learning techniques aiming to explore the positioning of different alternatives by an individual (@tsoukias2013ph, @pigozzi2016pai) and many other.

# Aims and objectives

The problematic arises from the insufficient points of contact among users (economists and engineers) and data scientists, who pursue different objectives, although using similar techniques.
A work that uses unified knowledge from both disciplines might be highly beneficial for researchers and provide support for future applied studies.
Following the logic of @athey2018iml and @mullainathan2017ml the project will attempt to merge the essentials of ML and Econometrics paradigms, retaining their key concepts.

First of all, as an interdisciplinary research project, the PhD work aims to define a common analysis and a comparison grid of performances of different models issued from econometrics and ML in application to individual decision making process. Definion of the performance criteria will have to take into account the specificity of the research objectives (data exploration, prediction, explanatory power, operational effectiveness, etc) and constraints of the two scientific domains.

A second objective is to incorporate the recent developments in the decision making theory into the general model comparison frameworks. This is a demanding procedure, that will require a thorough study of the discrete choice models in the context of consumer behaviour exploration, taking into account the underlying theoretical decision making theories.
An exploration of the eventual changes in models' performances depending on the underlying theory is of particular interest, taking into account the trends in the recommender systems (@sihem2009rs), preference learning (@pigozzi2016pai, @furnkranz2011p) and econometrics choice modelling (@joly2019qcm).

A third objective is to evaluate performance of these different modelling techniques in presence of specific behavioral bias in applied fields.

First, the *risk aversion* in investment choice: Different ways to model choices in presence of risk aversion (loss aversion, probability weighting, context-dependence and salience (@ODonoghue2018))  may be used to simulate choices and compare the capacity of the ML and econometrics tools to detect and/or to take into account these potential biases. 

Second, linear tariff and two-part tariff are proposed in many fields (energy consumption, transport,...) based on estimates of the demand response and consumer classification. Models performance in presence of *heterogeneous preferences* may be evaluated through simulated choices of contract. The efficient contract design for the simulated and controlled heterognenous population will be the benchmark of the contracts designed based on the estimations and predictions of the evaluated techniques.

Third, impact of the *Independance to Irrelevant Alternatives* (IIA) assumption of the 'work-horse' multinomial logit will be evaluated in the different modelling frameworks from ML and econometrics. Structures of the choice set will be studied, for example in transport fields, distinguishing air from terrestrial transport modes in long distance travel or mechanised vs non-mechanised mode in urban mobility. Following @joly2019qcm, extension could be towards questionning *reference dependance bias* in the context of mobility choices and  modelling.

# Proposed research methodology

A literature review will establish an extensive taxonomy of multiple basic discrete choice models issued from *Machine Learning* and *Econometrics* fields. The performance comparison grid will benefit from this taxonomy of techniques and understanding of the way the models perform (which is crucial to assess their performances under various theoretical specifications on the real world data). 

Assessment and correct comparison of the models theoretical performances may be achieved over a synthetic data with known predetermined and controlled properties. 
Construction of a synthetic dataset will follow techniques, described by @drechsler2011sd, @garrow2010gs or more recently by @kar2019ms.

Finally, verification of the findings on the real world data may benefit from controlled experiments to validate the theoretical findings, and provide further insights for the model comparison (specifically for the three fields of application: choice under uncertainty, choice of constracts and choice in hierarchical choice set)

# Expected research contribution

This work is expected to contribute to the interdisciplinary unification of very different domains, such as *Econometrics* and *Machine Learning* in the context of decision making modelling.
Introduction of developments in the decision making theory into the general model comparison frameworks will permit to study a wide range of statistical models as well as the underlying decision making theories in order to provide a complete overview of the existing cross-disciplinary methodology.

Crossing econometric simulation tools and economic experimentation will rely on the definition of a unified framework to test and validate behavioral assumption. A tool to design simultaneously a choice experiment and its associated simulated choice could benefit the research community to ease and improve the bridge between experimentation method and the corresponding data analysis method.

The valuable insights into model performance in the context of discrete individual choice, as well as underlying behavioural theories influences on the results are expected to be obtained. A detailed study will allow for future researchers to choose the appropriate method for a specific application in relation with cognitive and behavioural studies. The proposed applications may 1) improve demand analysis and consumer classification for contract design, 2) identify efficient tools to model and predict choices facing uncertainty and 3) evaluate impact of misspecified choice structure.

In terms of immediate results the PhD thesis will produce a methodological support, describing and putting in relations the different models specific to the field of application.  
