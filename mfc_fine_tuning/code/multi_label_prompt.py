PROMPT_MULTI = '''You are analyzing how texts are framed in their coverage of topics. 
Your task is to identify the frames used in the text into one of nine frame categories based on how the issue is presented and which aspects are emphasized. 

Instructions:
1. Read the news article carefully.
2. Identify the frame that shapes the article's narrative.
   Typically, only one frame applies, but if multiple frames are present, please select all that apply, separated by commas.
3. Output ONLY the numerical code for those frames first, followed by a new line.

Classify the frames used in the news article choosing numerical labels from the following list:
1 - Economic: Content highlighting financial implications, costs, benefits, economic growth, budget impacts, market effects, or monetary considerations.

3 - Moral: Content emphasizing ethical principles, religious perspectives, moral duties, values-based arguments, or appeals to righteousness.

4 - Fairness and Equality: Content focusing on equitable treatment, discrimination, rights balancing, justice, equal opportunity, or disparate impacts across different groups.

5 - Legal: Content focusing on laws, constitutionality, crime, enforcement, judicial interpretation, legal precedents, or violations of regulations. Includes both abstract legal principles and specific applications of laws.

6 - Political-Policy: Content emphasizing political processes, partisan considerations, policy implementation, legislative procedures, political strategy, or governance approaches.

8 - Security: Content highlighting protection from threats, safety measures, defense concerns, stability risks, or prevention of harm to individuals or communities.

9 - Health: Content centered on healthcare, disease, physical/mental wellbeing, public health outcomes, safety concerns, or health-related quality of life.

11 - Cultural Identity: Content examining social norms, cultural values, traditions, identity-based perspectives, or societal customs as they relate to the issue.

12 - Public Opinion: Content focusing on polling data, public sentiment, demographic attitudes, social trends, or shifts in collective viewpoints.


'''