import os
import sys
import uuid
import glob
import json
import openai
from openai import OpenAI
from PIL import Image, ImageSequence
import requests
from dotenv import load_dotenv
from pdf2image import convert_from_path
import base64

import copy
import pdfkit
from pypdf import PdfMerger, PdfWriter
from PyPDF2 import PdfReader
import numpy as np
from skimage import util
from skimage.io import imsave, imread

import random


load_dotenv()

POPPLER_PATH = os.environ['POPPLER_PATH']
OPENAI_KEY = os.environ['OPENAI_KEY']
TEMPLATE_STORE_PATH = os.environ['TEMPLATE_STORE_PATH']
WKHTMLTOPDF_PATH = os.environ['WKHTMLTOPDF_PATH']
PERSONA_PROMPT = f"you are an expert in document undertanding and synthetic data generation.\n\nFirst generate an individual from /country/ that works for /company_name/ in the /company_industry/ industry. return as plain text persona description."
PROMPT = f"NOTE - slight changes to the template style (specifically background and element/feature colors and iconography) and layout can be made to align it more closely to the country : /country/\n\nRULE - ALL numbers within this page MUST be unique whole numbers with no decimal places and contain at least one odd number\n\nyou are an expert in document undertanding and synthetic data generation. the persona provided will define all language/text (including characters etc), currency, locations, symbols and content returned within this page.\n\nThen based on the template provided generate a new page (translated to /country/) which MUST include all the data provided within the 'key values' (you may need to extend, alter or update the template to include the keys and values provided).\n\n. The final page must not include any place holder values or logos/images\n\nreturn as a styled utf8 encoded html (including different fonts which may include handwritten elements where appropriate). you must always include the following html tag : <meta http-equiv=\"content-type\" content=\"text/html; charset=utf-8\" />\n\npersona : /persona/\n"
KEY_CHECK_PROMPT = "you are an expert in data analysis, based on the html page data and key values provided, determine if each key value pair appears or is visible with minor changes to format or abbreviation (date can be considered a match even if their formats to not align) within the html data. return a JSON object with the format : {'used_keys':[<string> - list of all keys that appear in the html in the format (key - reasoning)], 'unused_keys':[<string> - list of all keys that do not appear in the html in the format (key - reasoning)]}"

config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ask_gpt(prompt, body, gptmodel="gpt-4o", image_path=None):
	client = OpenAI(api_key=OPENAI_KEY)
	try:
		if image_path != None:
			try:
				base64_image = encode_image(image_path)
				response = client.responses.create(
					model= gptmodel,
					input=[
						{
							"role": "user",
							"content": [
								{
									"type": "input_image",
									"image_url": f"data:image/png;base64,{base64_image}"
								},
								{
									"type": "input_text",
									"text": f"{prompt}\n{body}"
								}
							]
						}
					],
					text={
						"format": {
								"type": "text"
							}
						},
					reasoning={},
					tools=[],
					temperature=1,
					max_output_tokens=12000,
					top_p=1,
					store=True
				)

				return response.output_text            
			except Exception as ex:
				print(f"ERROR - GPT IMAGE : {ex}")
		
		response = client.responses.create(
      model= gptmodel,
      input=[
          {
              "role": "user",
              "content": [                
                  {
                    "type": "input_text",
                    "text": f"{prompt}\n{body}"
                  }
              ]
          }
      ],
      text={
          "format": {
            "type": "text"
          }
      },
      reasoning={},
      tools=[],
      temperature=1,
      max_output_tokens=14000,
      top_p=1,
      store=True
		)
		return response.output_text
	except Exception as ex:
		print(f"ERROR - GPT4 Error : {ex}")
		return ""

class SyntheticDocTemplate():
	_id = str(uuid.uuid4())
	template_path = None
	name = None
	type = None
	company_name = None
	company_industry = None
	number_pages = 0
	critical_fields_to_extract = []
	page_templates = []

	def __init__(self, _template_path=None, _name=None, _type=None, _company_name=None, _company_industry=None, _number_pages=0, _critical_fields_to_extract=None, _page_templates=[]):
		self._id = str(uuid.uuid4())
		self.name = _name
		self.type = _type
		self.company_name = _company_name
		self.company_industry = _company_industry
		self.number_pages = _number_pages
		self.critical_fields_to_extract = _critical_fields_to_extract
		self.page_templates = _page_templates
		self.template_path = _template_path	

	def load_from_json(self):
		if os.path.exists(self.template_path):			
			with open(self.template_path, 'r') as file:
				data = json.load(file)
				self._id = data['id']
				self.name = data['name']
				self.type = data['type']
				self.company_industry = data['company_industry']
				self.number_pages = data['number_pages']
				self.critical_fields_to_extract = data['critical_fields_to_extract']
				self.page_templates = data['page_templates']			
		else:
			print(f"WARNING - unable to locate template at : {self.template_path}")

	def show_template(self):
		data = {
			"id":self._id,
			"name":self.name,
			"type":self.type,
			"company_name":self.company_name,
			"company_industry":self.company_industry,
			"number_pages":self.number_pages,
			"critical_fields_to_extract":self.critical_fields_to_extract,
			"page_templates":self.page_templates
		}
		print(f"DEBUG - template : {data}")

	def save_template(self):
		data = {
			"id":self._id,
			"name":self.name,
			"type":self.type,
			"company_name":self.company_name,
			"company_industry":self.company_industry,
			"number_pages":self.number_pages,
			"critical_fields_to_extract":self.critical_fields_to_extract,
			"page_templates":self.page_templates
		}

		print(f"DEBUG - saving to template path : {self.template_path}")
		with open(self.template_path, 'w') as f:
			json.dump(data, f)

class SyntheticDocEngine():
	templates = []
	current_template = []
	doc_path = None
	fields_to_capture = None	
	template_name = None
	template_type = None
	doc_type = None
	contains_photo = False
	pages = []
	countries = []

	def add_noise(self, image_path):
		if random.randint(0,100) >= 40:
			print(f"DEBUG - adding noise : {image_path}")
			img = imread(image_path)
			img_gray = img
			noise_level = random.randint(50, 75)
			noise = np.random.normal(0, noise_level, img_gray.shape)
			img_noised = img_gray + noise
			img_noised = np.clip(img_noised, 0, 255).astype(np.uint8)
			imsave(image_path, img_noised)

	def convert_resp_to_json(self, gpt_response):
		if "```json" in str(gpt_response):
			gpt_response = str(gpt_response).split("```json")[1]
		if "```" in str(gpt_response):
			gpt_response = str(gpt_response).split("```")[0]

		json_response = {}
		try:
			json_response = json.loads(gpt_response)
		except:
			print(f"WARNING - failed to parse to json ASSUMING JSON")
			json_response = gpt_response
		return json_response

	def generate_template_data_fields(self, data_fields, template_type, country="United States", company_name="Best Wurst", company_industry="food / hospitality"):
		print(f"DEBUG - generating data fields for : {data_fields}")
		field_generation_prompt = "RULE - if a data field is marked as '_bool' then it is 'yes' or 'no' depending on if a related value exists.\n\nRULE - ALL numbers within this page MUST be unique whole numbers with no decimal places and contain at least one odd number\n\nRULE - ALL identifying numbers (account id, invoice id and similar) must be a mix of alphanumeric characters with no consecutive numbers or letters eg (123... or ABC...).\n\nyou are an expert in synthetic data generation. first pick an industry and sector at random, picking from the most common within that country (cannot be office supplies). then based on that selection generate example values for the provided data fields based on the provided country (which will effect language and characters) and company industry and return a complete JSON object {'key_values':[{'key':'<string - data field>','value':'<string - generated value>', 'key_description': '<string - description of key>'}]}\ncountry : /country/\n\ncompany industry : /company_industry/"
		body = f"data fields : {data_fields}"
		field_generation_prompt = field_generation_prompt.replace("/country/", country)
		field_generation_prompt = field_generation_prompt.replace("/company_industry/", company_industry)
		key_value_resp = ask_gpt(field_generation_prompt, body, gptmodel="gpt-4o-mini")	
		return self.convert_resp_to_json(key_value_resp)

	def generate_document_from_template(self, number_of_documents):
		print(f"DEBUG - generating : {number_of_documents} documents for synthetic template : {self.current_template._id} / {self.current_template.name}")
		self.current_template.show_template()
		for i in range(int(number_of_documents)):
			print("\n")
			country = str(self.countries[random.randint(0, (len(self.countries)-1))]).strip()			
			company_name = "acme"
			if self.current_template.company_name:
				company_name = self.current_template.company_name
			document_id = str(uuid.uuid4())
			persona_prompt = PERSONA_PROMPT.replace("/country/", country).replace("/document_type/", self.current_template.type).replace("/company_name/", company_name).replace("/company_industry/", self.current_template.company_industry)
			persona = ask_gpt(persona_prompt,"", gptmodel="gpt-4o-mini")	
			template_key_values = self.generate_template_data_fields([cf.replace("~",",") for cf in self.current_template.critical_fields_to_extract], self.current_template.type, country, company_name, self.current_template.company_industry)
			print(f"DEBUG - generated template key values : {template_key_values}")		
			# print(f"DEBUG - persona : {persona}")
			pdf_pages = []
			page_html_str_agg = ""
			skip_doc = False
			keys_present_in_doc = []
			for page_ix, page in enumerate(self.current_template.page_templates):		
				page_html = base64.b64decode(page['html_template'])
				# print(page_html)
				temp_decode_html_path = "tmp/html/html_decode_"+str(document_id)+"_"+str(page_ix)+".html"
				with open(temp_decode_html_path, 'w') as f:
					f.write(str(page_html).replace("b'","").replace("</html>\n'","</html>"))

				print(f"DEBUG : page : {page_ix} : {page['fields_to_capture']} / {len(page['fields_to_capture']['keys'])}")
				page_local_fields_to_generate = page['local_fields_to_generate']
				if len(page_local_fields_to_generate) >= 1:
					print(f"DEBUG - generating page specific keys : {page_local_fields_to_generate}")
					template_key_values_prompt = f"you are an expert in document understanding, based on the provided page html determine the type of data represented and then generate values for the provided keys {page_local_fields_to_generate} based on data in the page template. finally update and return the key values JSON provided with those generated values :\n\nkey_values : {template_key_values}\n\nhtml template : {page_html}"
					template_key_values = ask_gpt(template_key_values_prompt, "", gptmodel="gpt-4o-mini")
					template_key_values = str(template_key_values).split("```json")[1].split("```")[0]
					template_key_values = json.loads(template_key_values)	
					print(f"DEBUG - updated key values : {template_key_values}")
					print(f"\n\n\n{template_key_values_prompt}")
				

				# select the key values that apply
				page_kv_resp = ""
				if len(page['fields_to_capture']['keys']) >= 1:
					selected_page_key_values = "you are an expert in understanding json. first determine the type and content of this page based on the 'expected page keys'. then select only the key values from the provided 'template_key_values' that apply to this page based on the provided keys. return as JSON"
					page_kv_resp = ask_gpt(selected_page_key_values, f"template key values : {template_key_values}\nexpected page keys : {page['fields_to_capture']}")

					# print(f"DEBUG - page key values : \n{page['fields_to_capture']}\nselected : {page_kv_resp}")
					try:
						page_kv_resp = str(page_kv_resp).split("```json")[1]
						page_kv_resp = str(page_kv_resp).split("```")[0]
					except:
						pass
				
				gen_prompt = PROMPT.replace("/persona/", persona).replace("/country/", country)				
				generation_resp = ask_gpt(gen_prompt, f"key values : {page_kv_resp}\nhtml template : {page_html}")
				if "```html" in generation_resp:	
					raw_response = generation_resp.split("```html")[-1]
					raw_response = raw_response.split("```")[0]
				else:
					print(f"DEBUG - unable to determine if html: \n{generation_resp}")

				keys_used_resp = ask_gpt(KEY_CHECK_PROMPT, f"\nhtml data : {raw_response}\n key values : {page_kv_resp}")		
				keys_present_on_page = []
				try:
					keys_present_on_page = str(keys_used_resp).split("```json")[-1]
					keys_present_on_page = json.loads(str(keys_present_on_page).split("```")[0])
				except:
					keys_present_on_page = []
				
				print(f"DEBUG - keys used resp : {keys_present_on_page['used_keys']}")
				print(f"DEBUG - keys NOT used resp : {keys_present_on_page['unused_keys']}")
				print(f"DEBUG - keys used resp w/o reason : {[str(k).split('-')[0].strip() for k in keys_present_on_page['used_keys']]}")
				try:
					keys_present_in_doc.extend([str(k).split("-")[0].strip() for k in keys_present_on_page['used_keys']])
				except:
					pass

				page_html_str_agg += str(raw_response)
				html_path = "tmp/html/example_"+str(self.current_template.type).replace(' ','_')+"_"+str(document_id)+"_"+"_page_"+str(page_ix)+".html"
				text_file = open(html_path, "wb")
				text_file.write(raw_response.encode("utf-8"))
				text_file.close()
				pdf_path = "tmp/pdf/example_"+str(self.current_template.type).replace(' ','_')+"_"+str(document_id)+"_"+"_page_"+str(page_ix)+".pdf"
				try:
					pdfkit.from_file(html_path, pdf_path, configuration = config, options={'encoding': 'UTF-8','enable-local-file-access': True})
					pdf_pages.append(pdf_path)
				except Exception as ex:
					print(f"ERROR - generating page : {page_ix} : {ex}")
					if page_ix == 0:
						skip_doc = True 
						break

			try:
				print(f"DEBUG - final keys used on page : {keys_present_in_doc}")
				clean_template_key_values = [kv for kv in template_key_values['key_values'] if kv['key'] in keys_present_in_doc]
				template_key_values['key_values'] = clean_template_key_values
				print(f"DEBUG - re engineered kvs : {template_key_values}")				

			except Exception as ex:
				print(f"ERROR - in key value json : {template_key_values}\n{ex}")

			final_pdf_path = "tmp/pdf/FINAL_"+str(self.current_template.type).replace(' ','_')+"_"+str(country).replace(" ","_")+"_"+str(document_id)+".pdf"
			cropped_final_pdf_path = "tmp/pdf/FINAL_"+str(self.current_template.type).replace(' ','_')+"_"+str(country).replace(" ","_")+"_"+str(document_id)+".pdf"
			print(f"DEBUG - pages generated... finalising pdf : {final_pdf_path}")
			merger = PdfWriter()
			for page_ix, pdf in enumerate(pdf_pages):
				# if page_ix <= 1:
				merger.append(pdf)

			merger.write(final_pdf_path)
			merger.close()

			try:
				clean_template_key_values = {"key_values":[]}
				for kvitem in [kv for kv in template_key_values['key_values'] if "_bool" not in kv['key']]:
					associated_bool = [kv for kv in template_key_values['key_values'] if str(kvitem['key'].split(" - ")[0])+"_bool" == kv['key']]
					print(f"DEBUG - kv {kvitem} associated bool : {len(associated_bool)}")

					if len(associated_bool) >= 1:
						kvitem['value'] = associated_bool[0]['value']

					clean_template_key_values['key_values'].append({"key":kvitem['key'].split(" - ")[0], "value":kvitem['value']})

				template_key_values = clean_template_key_values

				testing_json_path = "tmp/testing_schema/FINAL_"+str(self.current_template.type).replace(' ','_')+"_"+str(country).replace(" ","_")+"_"+str(document_id)+".json"
				with open(testing_json_path, 'w') as f:
				    json.dump(template_key_values, f)

				print(f"DEBUG - pdf generated : {final_pdf_path}")
				print(f"DEBUG - testing schema generated : {testing_json_path}")
			except Exception as ex:
				print(f"ERROR - saving docs : {ex}")
				try:
					os.remove(cropped_final_pdf_path)
				except:
					pass
			

	def load_templates(self):
		files = glob.glob(str(TEMPLATE_STORE_PATH)+"/*")
		for f in files:
			if os.path.isfile(f):
				if "json" in f:					
					template = SyntheticDocTemplate(f)
					template.load_from_json()
					self.templates.append(template)
		print(f"DEBUG loaded : {len(self.templates)} templates")

	def list_templates(self):
		for template in self.templates:
			print(f"DEBUG - template : {template.name} - {template.type}  - {template.critical_fields_to_extract}")

	def set_template(self, template_index):
		self.current_template = self.templates[template_index]

	def gain_fields_based_on_document_type(self):
		prompt = "you are an expert in synthetic data generation with a deep understanding of business document data. generate a list of fields that are commonly extracted from the document : "+str(self.template_type)+".you must return a JSON object with the format {\"fields\":[\"<string - name of key>\"]}"
		gained_gpt_fields = ask_gpt(prompt, "")		
		if "```json" in gained_gpt_fields:
			gained_gpt_fields = str(gained_gpt_fields).split("```json")[1]
			gained_gpt_fields = str(gained_gpt_fields).split("```")[0]
			try:
				gained_gpt_fields = json.loads(gained_gpt_fields)				
				self.fields_to_capture = gained_gpt_fields['fields']				
			except Exception as ex:
				print(f"ERROR - generating field set... : {ex}")
				exit()
				pass

	def check_support(self):
		if self.doc_path:
			if ".pdf" in self.doc_path.lower():
				self.doc_type = "pdf"
			elif ".png" in self.doc_path.lower():
				self.doc_type = "png"
			elif ".jpg" in self.doc_path.lower() or ".jpeg" in self.doc_path.lower():
				self.doc_type = "jpg"
			else:
				print(f"ERROR - doc type not supported.")
				exit()		

	def gain_pages(self):
		if self.doc_type == None:
			self.pages.append('rawdocpage')
		elif self.doc_type == "pdf":
		    images = convert_from_path(self.doc_path, poppler_path=POPPLER_PATH)
		    for i in range(len(images)):
		        doc_name = "doc_page_"+str(i)+"_"+str(uuid.uuid4())
		        file_type = "image/png"
		        tmp_path = "./tmp/doc_page_"+str(i)+"_"+str(uuid.uuid4())+".png"
		        images[i].save(tmp_path, 'PNG')
		        self.pages.append(tmp_path)        
		else:
			self.pages.append(doc_path)

	def __init__(self, _doc_path=None, _fields_to_capture=None, _template_name=None, _template_type=None, _contains_photo=False):		
		self.doc_path = _doc_path
		try:
			self.fields_to_capture = _fields_to_capture.lower()
		except:
			self.fields_to_capture = None
		self.template_name = _template_name
		self.template_type = _template_type
		self.doc_type = None
		self.contains_photo = _contains_photo
		self.pages = []
		self.countries = []
		# incl_field_names = True

	def generate_template(self):
		self.check_support()

		if self.fields_to_capture == None:			
			self.gain_fields_based_on_document_type()			

		print(f"DEBUG - template fields to capture : {self.fields_to_capture}")				
		print("DEBUG - SDE Template Builder")
		print(f"DEBUG - doc path : {self.doc_path}")
		print(f"DEBUG - doc type : {self.doc_type}")
		self.gain_pages()

		new_template = SyntheticDocTemplate(f"{TEMPLATE_STORE_PATH}/{self.template_type}_{str(uuid.uuid4())}.json", self.template_name, self.template_type, None, None, 0, self.fields_to_capture, [])

		prompt = "you are an expert in synthetic data generation with a deep understanding of business document data. generate a company name and company industry based on the type : "+str(self.template_type)+".you must return a JSON object with the format {\"company_name\":\"<string - name of company>\", \"company_industry\":\"<string - name of industry>\"}"
		gained_gpt_company_data = ask_gpt(prompt, "")
		if "```json" in gained_gpt_company_data:
			gained_gpt_company_data = str(gained_gpt_company_data).split("```json")[1]
			gained_gpt_company_data = str(gained_gpt_company_data).split("```")[0]
			try:
				gained_gpt_company_data = json.loads(gained_gpt_company_data)
				new_template.company_name = gained_gpt_company_data['company_name']
				new_template.company_industry = gained_gpt_company_data['company_industry']		
			except Exception as ex:
				print(f"ERROR - generating company data... : {ex}")
				exit()

			for page_ix, page in enumerate(self.pages):
				print(f"\nDEBUG - converting page : {page_ix} / {page}")
				# prompt = "you are an expert in document processing and image analysis. convert the provided image into a template. you must preserve the layout, styles and all data (anonymised). return as html."
				html_resp = ""

				while "<html" not in str(html_resp).lower():
					if page == "rawdocpage":
						print(f"DEBUG - raw page : {page_ix} generation for : {self.template_type} photo id : {self.contains_photo}")
						prompt = ""
						if not self.contains_photo:
							prompt = f"you are an expert in document processing. based on the document type of : {self.template_type} and the extraction fields : {self.fields_to_capture} generate a highly detailed and realistic template (based on real documents of the same type, visually interesting). (using tables with invisible borders to define the layout), colours (background/foreground/elements defined by or commonly assocaited with the template type : {self.template_type}), styles and anonymised data. return as utf-8 html (which includes CSS within the html)."
						else:
							prompt = f"you are an expert in document processing. based on the document type of : {self.template_type} and the extraction fields : {self.fields_to_capture} generate a highly detailed and realistic template (based on real documents of the same type, visually interesting). this document must contain a photo of the person that it relates to use an svg image as a placeholder for the persons face. (using tables with invisible borders to define the layout), colours (background/foreground/elements defined by or commonly assocaited with the template type : {self.template_type}), styles and anonymised data. return as utf-8 html (which includes CSS within the html)."
						body = ""
						html_resp = ask_gpt(prompt, body)
					else:			
						prompt = f"you are an expert in document processing and image analysis. first analyse this page and then generate a highly detailed template based on its content (extracting style and color). You must preserve the layout (using tables with invisible borders to define the layout), colours (background/foreground/elements defined by or commonly assocaited with the template type : {self.template_type}), styles and anonymised data. return as utf-8 html (which includes CSS within the html, css to include colors)."						
						body = ""
						html_resp = ask_gpt(prompt, body, page)						

				prompt = "you are an expert in document processing and analysis. select only they keys from the provided fields_to_capture that apply to this specific page. you must return a JSON object with the format {\"keys\":[{\"key\":\"<string - name of key>\"}"
				selected_kv_resp = ask_gpt(prompt, f"fields_to_capture : {[str(f.split('-')[0]).strip() for f in self.fields_to_capture]}\nhtml template : {html_resp}")
				if "```json" in selected_kv_resp:
					selected_kv_resp = str(selected_kv_resp).split("```json")[1]
					selected_kv_resp = str(selected_kv_resp).split("```")[0]
					try:
						selected_kv_resp = json.loads(selected_kv_resp)
					except:
						pass

				print(f"DEBUG - key values that apply to this page : {selected_kv_resp}")
				# print(f"DEBUG - html resp : {html_resp}")

				if "```html" in html_resp:
					html_str = str(html_resp).split("```html")[1]
					html_str = html_str.split("```")[0]

					html_template_path = "tmp/html/template_html"+str(page_ix)+"_"+str(uuid.uuid4())+".html"
					with open(html_template_path, 'w') as f:
						f.write(html_str)

					html_b64data = base64.b64encode(html_str.encode("utf-8"))
					new_template.page_templates.append({"id":str(uuid.uuid4()),"page":page_ix,"html_template":str(html_b64data).replace("b'","").replace("'",""), "local_fields_to_generate": [], "fields_to_capture":selected_kv_resp})
				else:
					html_str = "<html"+str(html_resp).split("<html")[1]
					html_str = html_str.split("</html>")[0] + "</html>"
					html_b64data = base64.b64encode(html_str.encode("utf-8"))
					new_template.page_templates.append({"id":str(uuid.uuid4()),"page":page_ix,"html_template":str(html_b64data).replace("b'","").replace("'",""), "local_fields_to_generate": [], "fields_to_capture":selected_kv_resp})

			new_template.save_template()
			self.current_template = new_template


# example of how to initialise the engine
synthetic_doc_eng = SyntheticDocEngine(None, None, "vetrinary-prescription", _template_type="vetrinary-prescription", _contains_photo=False)
synthetic_doc_eng.load_templates()
synthetic_doc_eng.list_templates()
synthetic_doc_eng.set_template(2)
synthetic_doc_eng.countries = ["Japan"]
synthetic_doc_eng.generate_document_from_template(1)

# generate a new template
# synthetic_doc_eng.generate_template()
