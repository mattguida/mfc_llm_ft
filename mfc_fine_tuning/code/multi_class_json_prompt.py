PROMPT_MULTI = '''
You are analyzing how texts are framed in their coverage of topics. 
Your task is to identify the frame used in the text - the way the article structures its narrative and presents the issue. 

Instructions:
1. Read the text carefully
2. Identify the frame that shape the text's narrative
3. Output ONLY the numerical code for those frames, separated by commas (e.g., "6")
4. Do NOT add any text or explanation in your response.

Classify the frame used in the news article choosing numerical labels from the following list:
   1 - Economic: The costs, benefits, or monetary/financial implications of the issue (to an individual, family, community, or to the economy as a whole). Example: Stories discussing revenue gains/losses or industry growth.,
   
   2 - Capacity and Resources: The lack of or availability of physical, geographical, spatial, human, and financial resources, or the capacity of existing systems and resources to implement or carry out policy goals. Although similar to the economic frame, this frame stresses the limitation of funding/time/resources/etc.,
   
   3 - Morality: Any perspective—or policy objective or action (including proposed action)—that is compelled by religious doctrine or interpretation, duty, honor, righteousness or any other sense of ethics or social responsibility.,
   
   4 - Fairness and Equality: Equality or inequality with which laws, punishment, rewards, and resources are applied or distributed among individuals or groups. Also the balance between the rights or interests of one individual or group compared to another individual or group.,
   
   5 - Legality, Constitutionality, and Jurisprudence: The constraints imposed on or freedoms granted to individuals, government, and corporations via the Constitution, Bill of Rights and other amendments, or judicial interpretation. This deals specifically with the authority of government to regulate, and the authority of individuals/corporations to act independently of government. It relates to a law, and whether or not that law was broken.,
   
   6 - Policy Prescription and Evaluation: Particular policies proposed for addressing an identified problem, and figuring out if certain policies will work, or if existing policies are effective. It relates to HOW a policy should work.,
   
   7 - Crime and Punishment: Specific policies in practice and their enforcement, incentives, and implications. Includes stories about enforcement and interpretation of laws by individuals and law enforcement, breaking laws, loopholes, fines, sentencing and punishment. Increases or reductions in crime. It relates to the actual application of a rule.,
   
   8 - Security and Defense: Security, threats to security, and protection of one's person, family, in-group, nation, etc. Generally an action or a call to action that can be taken to protect the welfare of a person, group, nation sometimes from a not yet manifested threat. Similar to the Health and Safety frame, but differs because it addresses a preemptive action to stop a threat from occurring.,
   
   9 - Health and Safety: Healthcare access and effectiveness, illness, disease, sanitation, obesity, mental health effects, prevention of or perpetuation of gun violence, infrastructure and building safety.,
   
   10 - Quality of Life: The effects of a policy on individuals' wealth, mobility, access to resources, happiness, social structures, ease of day-to-day routines, quality of community life, etc.,
   
   11 - Cultural Identity: The social norms, trends, values and customs constituting culture(s), as they relate to a specific policy issue.,
   
   12 - Public Opinion: References to general social attitudes, polling and demographic information, as well as implied or actual consequences of diverging from or "getting ahead of" public opinion or polls.,
   
   13 - Political: Any political considerations surrounding an issue. Issue actions or efforts or stances that are political, such as partisan filibusters, lobbyist involvement, bipartisan efforts, deal-making and vote trading, appealing to one's base, mentions of political maneuvering. Explicit statements that a policy issue is good or bad for a particular political party.,
   
   14 - External Regulation and Reputation: The United States' external relations with another nation; the external relations of one state with another; or relations between groups. This includes trade agreements and outcomes, comparisons of policy outcomes or desired policy outcomes. It includes trade agreements and outcomes, the perception of a nation/state or a group by another, border relations.,
   
Requirements:
- Use ONLY ONE numerical number for the classification in your response (e.g. "12")
- Do NOT add any text or explanation.
- Do NOT repeat or include the input text in your response.


Output schema:
{{
    "sentence": # The original text to analyze
    "label": <integer>  # Frame classification label
}}

What is the frame of the following text?
'''


PROMPT_5_LABELS = '''
You are analyzing how texts are framed in their coverage of topics. 
Your task is to identify the frame used in the text - the way the article structures its narrative and presents the issue. 

Instructions:
1. Read the text carefully
2. Identify the frame that shape the text's narrative
3. Output ONLY the numerical code for those frames, separated by commas (e.g., "6")
4. Do NOT add any text or explanation in your response.

Classify the frame used in the news article choosing numerical labels from the following list:
   1 - Economic: The costs, benefits, or monetary/financial implications of the issue (to an individual, family, community, or to the economy as a whole). Example: Stories discussing revenue gains/losses or industry growth.,
   
   5 - Legality, Constitutionality, and Jurisprudence: The constraints imposed on or freedoms granted to individuals, government, and corporations via the Constitution, Bill of Rights and other amendments, or judicial interpretation. This deals specifically with the authority of government to regulate, and the authority of individuals/corporations to act independently of government. It relates to a law, and whether or not that law was broken.,
   
   6 - Policy Prescription and Evaluation: Particular policies proposed for addressing an identified problem, and figuring out if certain policies will work, or if existing policies are effective. It relates to HOW a policy should work.,
   
   7 - Crime and Punishment: Specific policies in practice and their enforcement, incentives, and implications. Includes stories about enforcement and interpretation of laws by individuals and law enforcement, breaking laws, loopholes, fines, sentencing and punishment. Increases or reductions in crime. It relates to the actual application of a rule.,
   
   13 - Political: Any political considerations surrounding an issue. Issue actions or efforts or stances that are political, such as partisan filibusters, lobbyist involvement, bipartisan efforts, deal-making and vote trading, appealing to one's base, mentions of political maneuvering. Explicit statements that a policy issue is good or bad for a particular political party.,
   
Requirements:
- Use ONLY ONE numerical number for the classification in your response
- Do NOT add any text or explanation.
- Do NOT repeat or include the input text in your response.


Output schema:
{{
    "sentence": # The original text to analyze
    "label": <integer>  # Frame classification label
}}

What is the frame of the following text?

'''

FIND_TOPIC = '''
Please analyze the following article to determine if it substantively discusses the topic of gay marriage (also known as same-sex marriage or marriage equality).
Report 1 in your response if the article substantively discusses the topic of gay marriage, or 0 if it does not. Do not add any explanation or text.

Output schema:
{{
    "article_id": {article_id}, # the article ID of the news article
    "label": <integer>  # 1 if article about gay marriage otherwise 0
}}

Is the following article about gay marriage? 
'''