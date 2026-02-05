"""
Prompt templates for OC_SOD
"""

saliency_preference_prompt_en = """
        - Your previous saliency_reasoning is as follows:
        {reasoning}
        - The names of all objects in the following analysis need to be consistent with the object_short in reasoning.
            - object_short name : {obj_name_list}
        - You need to conduct a saliency preference analysis of the objects in the image, generating group portraits and preference descriptions for those who prefer the objects. The objects marked with a red border are the most salient ones.
        - Place the ordinary salient objects in the "object" key, and the salient objects within the red border in the "most_salient" key. Note that there may be multiple most salient objects.
        - Requirements for group portraits and preference descriptions:
            - The generated group portraits and preference pairs need to be as diverse as possible, generating 2-3 preferences for each object.
            - Requirements for group portraits:
                - The description of a single group portrait can include one or more dimensions such as age, occupation, and interests.
                - Age should not be too specific, using descriptions such as children, teenagers, young adults, middle-aged, elderly, etc.
                - Occupations can be described more broadly, such as students, teachers, doctors, engineers, drivers, artists, writers, journalists, emergency responders, etc.
                - Interests can be described more broadly, such as technology enthusiasts, sports enthusiasts, art enthusiasts, travel enthusiasts, outdoor sports enthusiasts, etc.
            - Requirements for preference descriptions:
                - The preference cannot directly describe the object, but rather a general preference inferred from the object.
                - Descriptive phrases can include "more sensitive to...", "more inclined to...", etc.
            - The names in the following "object" must be consistent with "object_short" in reasoning.
            - Except for reasoning, other content should not mention the red border.
            - is_union indicates whether the multiple objects in most_salient are of the same category or have a subordinate relationship. If they are, it should be "true"; otherwise, it should be "empty".
            - The combination of group portraits and preference descriptions needs to uniquely match the object, and should not lead to a single group portrait and preference pair corresponding to multiple objects in the image.
        - Use JSON format for output, with the format and examples as follows:
                {{
                    "reasoning": "Your thought process",
                    "preferences":[
                        {{
                            "most_salient":
                              {{
                                "object":["name of object1 consist with the object_short in saliency_reasoning","object2 if exists in the red border"],
                                "collective_name":"If there are multiple, give a general term, otherwise empty",
                                "is_union":"true or empty"
                              }}, 
                                "preferences": [{{"portrait":"Description of group portrait","preference":"Preference description"}},...] 
                              }},
                        {{
                            "object": "normal saliency object name consist with the object_short in saliency_reasoning", 
                            "preferences": [
                                {{
                                    "portrait":"Professionals engaged in emergency rescue or news reporting",
                                    "preference":"Highly sensitive to individual behavior in crisis situations"
                                }}, {{
                                    "portrait":"Middle-aged audience with strong empathy",
                                    "preference":"Easily empathizes with individuals in distress"
                                }},...]
                        }},...
                    ]
"""

saliency_reasoning_prompt_en = f"""
        - Saliency refers to certain regions in an image that stand out visually from their surroundings due to features such as color, contrast, texture, or motion. These regions typically attract the viewer's attention. Please objectively analyze the salient objects in the current image comprehensively.
        - Do not analyze at the component level of the object, such as "the screen of the computer", "the tire of the car", "the eye of the bird", etc.
        - Use JSON format for output, with the format and examples as follows:
                {{
                    "reasoning": "Your thought process",
                    "saliency_objects": [
                        {{
                            "object_referring":"the TV on the left wall",
                            "object_short": "TV",
                            "reason": "In terms of color and contrast, the brightness and colors of the TV screen differ significantly from the surrounding environment, making it easy to attract attention."
                        }},
                        {{
                            "object_referring":"the sofa in the center of the room",
                            "object_short": "sofa",
                            "reason": "The color and texture of the sofa contrast with the floor and walls, making it easy to notice."
                        }},
                    ],
                    "not_saliency_objects": [
                        {{
                            "object_referring":"the wall behind the TV",
                            "object_short": "wall",
                            "reason": "The color and texture of the wall are consistent with the overall environment, making it less likely to attract attention."
                        }},
                        {{
                            "object_referring":"the floor beneath the sofa",
                            "object_short": "floor",
                            "reason": "The color and texture of the floor are consistent with the overall environment, making it less likely to attract attention."
                        }},
                    ]
                }}
        """

human_centric_intent_single_obj_prompt_en = """
        - Based on the salient object identified by the red border in the image, construct pairs of intent and analysis, where the intent is an implicit task-oriented intention, and the analysis is the reasoning for completing that intention.
            - The name of the unique salient object: {obj_name}
            - The red border is not part of the original image, it is only a tool to
                mark the salient object. The following intent and analysis should not mention the red border.
        - Requirements for generating intent:
            - The intent needs to be sufficiently implicit and have interactive significance. It should not be a
                direct or indirect reference to the focused object. The level of implicitness should refer to the JSON example at the end.
                - This direct and indirect reference includes using general terms to describe the object, such as "this plant", "this device", "this decoration", etc.
            - The content of the intent needs to focus on the salient object, rather than other objects
                and should be able to connect with the phrase "Where would I be more inclined to direct my gaze?", but this phrase should not appear in the intent.
            - In outdoor scenes, first analyze your current location and state, and then based on that location, state, and the given object in the image, conceive reasonable and practical implicit intents.
            - In outdoor scenes, unless there are obvious subjective perspective cues, assume you are a pedestrian. The intent should not infringe on others' privacy and private property.
            - Note that you are the observer of the image, do not put your own intent into the perspective of the objects or characters in the image.
            - Construct intents that have unique saliency matching for the salient object, especially when there are similar objects in the image. The intent should not correspond to multiple objects in the image.
        - Requirements for generating analysis:
            - The focused object in the analysis can only be the current unique salient object.
            - The content of the analysis includes the focused object needed to complete the intent and the reason for focusing on it.
            - When outputting the analysis, use <obj></obj> tags to wrap the focused object in the list of focused objects, and the object name inside the tag should be consistent with the unique salient object name.
            - object_referring is the referring description of the focused object in the analysis, and object_short_name is the short name of the object consistent with that inside <obj></obj>.
        - Generate 3-4 pairs of intent and analysis that are practical and diverse for the current scene.
        - Multiple pairs of intent and analysis need to focus on the same given salient object. When the given salient object consists of multiple similar objects, analyze its saliency as a whole and do not focus on a part of it.
        - Requirements for the reasoning key:
            - The content of the reasoning includes:
                1. First identify the overall state of all objects in the current scene, and determine your current state and the state of the salient object.
                2. Determine the following aspects of the salient object within the red border:
                    - Authenticity (whether it is a non-real object such as a shadow that cannot be interacted with)
                    - Integrity (whether the object is complete enough to correctly judge appropriate interactions)
                    - Interactivity (whether the current observer's perspective can interact with the object)
                    If any of the above three points are not met, then the object has no significance for being focused on, and the intent_reasoning and object should be set to None.
                3. If there is significance for interaction, then based on the given salient object in the image, conceive various reasons for it to be focused on that distinguish it from other objects in the image, and think about constructing practical implicit intents.
                4. Reflect on whether the generated intent meets the requirements from two perspectives:
                  - Implicitness: There should be no direct reference to the focused object, including using general terms to describe the object, such as "this ornament", "this device", "this decoration", etc.
                  - Unique saliency matching: Whether there are situations where the intent can correspond to multiple objects in the image, especially when there are similar objects to the salient object.
        - Use JSON format for output, with the format and examples of intent and analysis that meet the requirements as follows:
         {{
              "reasoning": "Your thought process",
              "intent_analysis_pairs: [
                {{
                    "object_short_name": "laptop",
                    "object_referring": "The black laptop on the ground in the right front",
                    "intent": "When reading, I want to supplement or confirm some additional information",
                    "analysis": "<obj>laptop</obj> on the ground in the right front can connect to the internet, so I will first notice it and use it to search for additional information that needs to be supplemented or confirmed."
                }},
                {{
                    "object_short_name": "cat",
                    "object_referring": "referring description of the cat",
                    "intent": "I plan to clean up the desktop environment for more efficient work",
                    "analysis": "<obj>cat</obj> is blocking the laptop, so before using the laptop for work, I will first notice it and pick it up.",
                }},
                {{
                    "object_short_name": "computer's touchpad",
                    "object_referring": "referring description of the computer's touchpad",
                    "intent": "I plan to turn off the computer to save power",
                    "analysis": "<obj>computer's touchpad</obj> can be used to move the cursor to turn off the computer, so I will first notice it.",
                }}
                 ...
                  ]
            }}
"""
