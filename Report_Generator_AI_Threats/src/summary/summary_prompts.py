from openai import OpenAI


def topic_summary_by_company(text_to_analyze, model, number_of_reports, api_key, topic, entity_name, focus):

    if focus == 'risk':

        prompt = (
            f"""You are an expert market analyst tasked with summarizing the key aspects of {topic} related to {entity_name} based on a set of daily news reports.

            Your goal is to analyze the provided text and summarize the key elements related to {entity_name} in the context of {topic}. 
            Your summary must focus only on the risks for {entity_name}.

        Text Description:
        - The input text includes {number_of_reports} market reports, each explicitly structured with its Headline and Text. Each report is clearly separated by a '--- Report End ---' marker.

        Instructions:
            - **Summarize the text:**  Provide a concise yet detailed (4 sentences maximum) summary of how {topic} affects {entity_name} across all reports.
            - Focus only on the risks for {entity_name}. 
            - Focus exclusively on elements in the input text that explicitly mention direct implications for {entity_name}'s business operations.
            - Ignore any general predictions regarding the economy or other businesses. 
            - Focus exclusively on elements related to {topic}.
            - Use only the information provided in the text. Do NOT infer from outside knowledge.
            - Include specific dates, names, and figures mentioned in the text whenever possible.
        
        Output:
        **Summary of the key aspects:** A comprehensive summary highlighting key aspects related to {topic} that impact {entity_name}.

        Format:
        Return the result as a JSON object with the following structure:
        {{ "summary": "Your summary here"}}
        """
        )

    else:

        # Create the prompt incorporating the topic
        prompt = (
            f"""You are an expert market analyst tasked with summarizing the key aspects of {topic} related to {entity_name} based on a set of daily news reports.

            Your goal is to analyze the provided text and summarize the key elements related to {entity_name} in the context of {topic}.

        Text Description:
        - The input text includes {number_of_reports} market reports, each explicitly structured with its Headline and Text. Each report is clearly separated by a '--- Report End ---' marker.

        Instructions:
            - **Summarize the text:**  Provide a concise yet detailed (4 sentences maximum) summary of how {topic} affects {entity_name} across all reports. 
            - Focus exclusively on elements in the input text that explicitly mention direct implications for {entity_name}'s business operations.
            - Ignore any general predictions regarding the economy or other businesses. 
            - Focus exclusively on elements related to {topic}.
            - Use only the information provided in the text. Do NOT infer from outside knowledge.
            - Include specific dates, names, and figures mentioned in the text whenever possible.
        
        Output:
        **Summary of the key aspects:** A comprehensive summary highlighting key aspects related to {topic} that impact {entity_name}.

        Format:
        Return the result as a JSON object with the following structure:
        {{ "summary": "Your summary here"}}
        """
        )

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Send request to OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": text_to_analyze
            }
        ],
        response_format={ "type":"json_object"},
        temperature=0
    )
    
    # Extract and process the response
    summary = response.model_dump()

    try:
        summary = eval(summary['choices'][0]['message']['content'])['summary']
    except:
        summary = ''
    
    return summary

