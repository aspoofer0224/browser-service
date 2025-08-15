#!/usr/bin/env python3
"""
Job Agent - Automated job application using browser-use library
Applies for jobs using CV information from cv.pdf file
"""

import asyncio
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from browser_use import Agent, BrowserProfile, Controller
from browser_use.agent.views import ActionResult
from browser_use.browser.session import BrowserSession
from browser_use.llm import BaseChatModel
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from browser_use.llm.openai.serializer import OpenAIMessageSerializer
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.exceptions import ModelProviderError
from dataclasses import dataclass
from typing import TypeVar, overload
from collections.abc import Mapping
import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.shared_params.response_format_json_schema import JSONSchema, ResponseFormatJSONSchema

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatDeepSeek(BaseChatModel):
	"""
	A custom DeepSeek chat model that implements the BaseChatModel protocol.
	
	This class provides a clean interface for DeepSeek API while maintaining
	compatibility with the browser-use framework.
	"""
	
	# Model configuration
	model: str = "deepseek-chat"
	
	# Model params
	temperature: float | None = None
	
	# Client initialization parameters
	api_key: str | None = None
	base_url: str = "https://api.deepseek.com"
	timeout: float | httpx.Timeout | None = None
	max_retries: int = 10
	default_headers: Mapping[str, str] | None = None
	default_query: Mapping[str, object] | None = None
	http_client: httpx.AsyncClient | None = None
	_strict_response_validation: bool = False
	
	@property
	def provider(self) -> str:
		return 'deepseek'
	
	@property
	def name(self) -> str:
		return self.model
	
	def _get_client_params(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary for DeepSeek API."""
		base_params = {
			'api_key': self.api_key,
			'base_url': self.base_url,
			'timeout': self.timeout,
			'max_retries': self.max_retries,
			'default_headers': self.default_headers,
			'default_query': self.default_query,
			'_strict_response_validation': self._strict_response_validation,
		}
		
		# Create client_params dict with non-None values
		client_params = {k: v for k, v in base_params.items() if v is not None}
		
		# Add http_client if provided
		if self.http_client is not None:
			client_params['http_client'] = self.http_client
		
		return client_params
	
	def get_client(self) -> AsyncOpenAI:
		"""Returns an AsyncOpenAI client configured for DeepSeek API."""
		client_params = self._get_client_params()
		return AsyncOpenAI(**client_params)
	
	def _get_usage(self, response: ChatCompletion) -> ChatInvokeUsage | None:
		"""Extract usage information from DeepSeek response."""
		if response.usage is not None:
			return ChatInvokeUsage(
				prompt_tokens=response.usage.prompt_tokens,
				prompt_cached_tokens=None,  # DeepSeek doesn't provide cached tokens
				prompt_cache_creation_tokens=None,  # DeepSeek doesn't provide cache creation tokens
				prompt_image_tokens=None,  # DeepSeek doesn't provide separate image token count
				completion_tokens=response.usage.completion_tokens,
				total_tokens=response.usage.total_tokens,
			)
		return None
	
	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...
	
	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...
	
	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""
		Invoke the DeepSeek model with the given messages.
		
		Args:
			messages: List of chat messages
			output_format: Optional Pydantic model class for structured output
			
		Returns:
			Either a string response or an instance of output_format
		"""
		
		openai_messages = OpenAIMessageSerializer.serialize_messages(messages)
		
		try:
			if output_format is None:
				# Return string response
				response = await self.get_client().chat.completions.create(
					model=self.model,
					messages=openai_messages,
					temperature=self.temperature,
				)
				
				usage = self._get_usage(response)
				return ChatInvokeCompletion(
					completion=response.choices[0].message.content or '',
					usage=usage,
				)
			
			else:
				response_format: JSONSchema = {
					'name': 'agent_output',
					'strict': True,
					'schema': SchemaOptimizer.create_optimized_json_schema(output_format),
				}
				
				# Return structured response
				response = await self.get_client().chat.completions.create(
					model=self.model,
					messages=openai_messages,
					temperature=self.temperature,
					response_format=ResponseFormatJSONSchema(json_schema=response_format, type='json_schema'),
				)
				
				if response.choices[0].message.content is None:
					raise ModelProviderError(
						message='Failed to parse structured output from DeepSeek response',
						status_code=500,
						model=self.name,
					)
				
				usage = self._get_usage(response)
				parsed = output_format.model_validate_json(response.choices[0].message.content)
				
				return ChatInvokeCompletion(
					completion=parsed,
					usage=usage,
				)
		
		except RateLimitError as e:
			error_message = e.response.json().get('error', {})
			error_message = (
				error_message.get('message', 'Unknown model error') if isinstance(error_message, dict) else error_message
			)
			raise ModelProviderError(
				message=error_message,
				status_code=e.response.status_code,
				model=self.name,
			) from e
		
		except APIConnectionError as e:
			raise ModelProviderError(message=str(e), model=self.name) from e
		
		except APIStatusError as e:
			try:
				error_message = e.response.json().get('error', {})
			except Exception:
				error_message = e.response.text
			error_message = (
				error_message.get('message', 'Unknown model error') if isinstance(error_message, dict) else error_message
			)
			raise ModelProviderError(
				message=error_message,
				status_code=e.response.status_code,
				model=self.name,
			) from e
		
		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e


class JobData(BaseModel):
	"""Job data structure from jobsfetch.json"""
	id: int
	url: str
	title: str
	description: str
	company_name: str
	countries: list[str | None] = []
	cities: list[str | None] = []
	is_remote: bool
	salary: str = ""
	seniority: str = ""
	hours: str = ""
	created_at: str


class JobApplication(BaseModel):
	"""Job application data structure"""
	job_id: int
	job_title: str
	company: str
	location: str
	application_date: str = Field(default_factory=lambda: datetime.now().isoformat())
	job_url: str
	status: str = "applied"
	notes: str = ""


class JobSearchConfig(BaseModel):
	"""Configuration for job search parameters"""
	keywords: list[str] = ["python", "software engineer", "developer"]
	locations: list[str] = ["remote", "hybrid"]
	experience_level: str = "mid"
	job_type: str = "full-time"
	salary_min: int | None = None
	max_applications: int = 5
	target_sites: list[str] = ["linkedin.com", "indeed.com", "glassdoor.com"]


class JobAgent:
	"""Main job agent class for automated job applications"""
	
	def __init__(
		self,
		cv_path: str = "cv.pdf",
		deepseek_api_key: str | None = None,
		headless: bool = False,
		config: JobSearchConfig | None = None,
		jobs_file: str = "jobsfetch.json"
	):
		self.cv_path = Path(cv_path)
		self.config = config or JobSearchConfig()
		self.jobs_file = Path(jobs_file)
		self.applications: list[JobApplication] = []
		self.available_jobs: list[JobData] = []
		self.logs_dir = Path("job_agent_logs")
		self.logs_dir.mkdir(exist_ok=True)
		
		# Set up logging first
		logging.basicConfig(
			level=logging.INFO,
			format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
			handlers=[
				logging.FileHandler(self.logs_dir / f"job_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
				logging.StreamHandler()
			]
		)
		self.logger = logging.getLogger(__name__)
		
		# Extract CV text at initialization
		self.cv_text = self._extract_cv_text()
		
		# Initialize LLM
		self.llm = ChatDeepSeek(
			model='deepseek-chat',
			api_key=deepseek_api_key,
			temperature=0.1
		)
		
		# Browser profile with job site access
		self.browser_profile = BrowserProfile(
			headless=headless,
			user_data_dir=str(self.logs_dir / "browser_profile"),
			window_size={'width': 1920, 'height': 1080}
		)
		
		# Custom controller for job-specific actions
		self.controller = Controller()
		self._register_custom_actions()
		
		# Load jobs from JSON file
		self._load_jobs_from_file()
	
	def _extract_cv_text(self) -> str:
		"""Extract text from CV PDF file using pdftotext"""
		if not self.cv_path.exists():
			self.logger.error(f"CV file not found: {self.cv_path}")
			return ""
		
		try:
			# Try using pdftotext first
			result = subprocess.run(
				['pdftotext', str(self.cv_path), '-'],
				capture_output=True,
				text=True,
				timeout=30
			)
			
			if result.returncode == 0:
				cv_text = result.stdout.strip()
				self.logger.info(f"Successfully extracted CV text ({len(cv_text)} characters)")
				return cv_text
			else:
				self.logger.warning(f"pdftotext failed with return code {result.returncode}")
				
		except subprocess.TimeoutExpired:
			self.logger.error("CV text extraction timed out")
		except FileNotFoundError:
			self.logger.warning("pdftotext not found, trying alternative methods")
			
		# Fallback: Try using Python libraries
		try:
			import PyPDF2
			with open(self.cv_path, 'rb') as file:
				reader = PyPDF2.PdfReader(file)
				text = ""
				for page in reader.pages:
					text += page.extract_text()
				
				if text.strip():
					self.logger.info(f"Successfully extracted CV text using PyPDF2 ({len(text)} characters)")
					return text.strip()
				else:
					self.logger.warning("PyPDF2 extracted empty text")
					
		except ImportError:
			self.logger.warning("PyPDF2 not available")
		except Exception as e:
			self.logger.error(f"PyPDF2 extraction failed: {e}")
		
		# Final fallback: Try pdfplumber
		try:
			import pdfplumber
			with pdfplumber.open(self.cv_path) as pdf:
				text = ""
				for page in pdf.pages:
					text += page.extract_text() or ""
				
				if text.strip():
					self.logger.info(f"Successfully extracted CV text using pdfplumber ({len(text)} characters)")
					return text.strip()
				else:
					self.logger.warning("pdfplumber extracted empty text")
					
		except ImportError:
			self.logger.warning("pdfplumber not available")
		except Exception as e:
			self.logger.error(f"pdfplumber extraction failed: {e}")
		
		self.logger.error("All CV text extraction methods failed")
		return ""
	
	def _register_custom_actions(self):
		"""Register custom actions for job applications"""
		
		@self.controller.action("Extract job details from current page")
		async def extract_job_details(browser_session: BrowserSession) -> ActionResult:
			"""Extract job information from the current page"""
			try:
				page = await browser_session.get_current_page()
				
				# Generic selectors that work across different job sites
				job_title = await page.locator('h1, [data-testid*="job-title"], .job-title, .jobTitle').first.text_content()
				company = await page.locator('[data-testid*="company"], .company, .companyName, .employer').first.text_content()
				location = await page.locator('[data-testid*="location"], .location, .jobLocation').first.text_content()
				
				job_data = {
					"job_title": job_title or "Unknown",
					"company": company or "Unknown", 
					"location": location or "Unknown",
					"job_url": page.url
				}
				
				return ActionResult(
					extracted_content=json.dumps(job_data),
					success=True
				)
			except Exception as e:
				self.logger.error(f"Failed to extract job details: {e}")
				return ActionResult(
					extracted_content=json.dumps({"error": str(e)}),
					success=False
				)
		
		@self.controller.action("Upload CV file to file input")
		async def upload_cv(index: int, browser_session: BrowserSession) -> ActionResult:
			"""Upload CV file to a file input element"""
			try:
				if not self.cv_path.exists():
					return ActionResult(
						extracted_content="CV file not found",
						success=False
					)
				
				page = await browser_session.get_current_page()
				file_inputs = await page.locator('input[type="file"]').all()
				
				if index < len(file_inputs):
					await file_inputs[index].set_input_files(str(self.cv_path))
					return ActionResult(
						extracted_content="CV uploaded successfully",
						success=True
					)
				else:
					return ActionResult(
						extracted_content="File input not found at specified index",
						success=False
					)
			except Exception as e:
				self.logger.error(f"Failed to upload CV: {e}")
				return ActionResult(
					extracted_content=f"Upload failed: {str(e)}",
					success=False
				)
	
	def _load_jobs_from_file(self):
		"""Load jobs from the jobsfetch.json file"""
		if not self.jobs_file.exists():
			self.logger.error(f"Jobs file not found: {self.jobs_file}")
			return
		
		try:
			with open(self.jobs_file, 'r') as f:
				data = json.load(f)
			
			# Handle both direct list and nested structure
			if isinstance(data, dict) and "hits" in data:
				jobs_data = data["hits"]
			else:
				jobs_data = data
			
			for job_data in jobs_data:
				try:
					job = JobData(**job_data)
					self.available_jobs.append(job)
				except Exception as e:
					self.logger.warning(f"Failed to parse job data: {e}")
					continue
			
			self.logger.info(f"Loaded {len(self.available_jobs)} jobs from {self.jobs_file}")
			
		except Exception as e:
			self.logger.error(f"Failed to load jobs from file: {e}")
			return
	
	
	def filter_jobs(self, max_jobs: int | None = None) -> list[JobData]:
		"""Filter available jobs based on config criteria"""
		filtered_jobs = []
		
		for job in self.available_jobs:
			# Check if job matches keywords
			keyword_match = False
			for keyword in self.config.keywords:
				if keyword.lower() in job.title.lower() or keyword.lower() in job.description.lower():
					keyword_match = True
					break
			
			if not keyword_match:
				continue
			
			# Check location preferences
			location_match = False
			for location in self.config.locations:
				if location.lower() == "remote" and job.is_remote:
					location_match = True
					break
				for city in job.cities:
					if city and location.lower() in city.lower():
						location_match = True
						break
				for country in job.countries:
					if country and location.lower() in country.lower():
						location_match = True
						break
			
			if location_match:
				filtered_jobs.append(job)
		
		if max_jobs:
			return filtered_jobs[:max_jobs]
		return filtered_jobs
	
	
	async def apply_to_job(self, job_data: JobData) -> JobApplication | None:
		"""Apply to a specific job posting"""
		self.logger.info(f"Applying to job: {job_data.title} at {job_data.company_name}")
		
		# Format location info
		valid_cities = [city for city in job_data.cities if city is not None]
		location_info = ", ".join(valid_cities) if valid_cities else "Remote" if job_data.is_remote else "Unknown"
		
		# Create comprehensive task description with CV context
		cv_context = f"\n\nMY CV INFORMATION:\n{self.cv_text}" if self.cv_text else "\n\nWarning: CV text extraction failed. Please use the CV file directly."
		
		task_description = f"""
		Navigate to {job_data.url} and apply for the job position: {job_data.title} at {job_data.company_name}.
		
		Job Details:
		- Title: {job_data.title}
		- Company: {job_data.company_name}
		- Location: {location_info}
		- Seniority: {job_data.seniority}
		- Type: {job_data.hours}
		- Job Description: {job_data.description[:500]}...
		{cv_context}
		
		Steps to follow:
		1. Navigate to the job URL
		2. Look for an "Apply" or "Apply Now" button and click it
		3. Fill out the application form using information from my CV above
		4. Upload the CV file when prompted (file path: {self.cv_path})
		5. Answer any screening questions based on my skills and experience from the CV
		6. For salary expectations, be conservative and research-based
		7. Submit the application
		8. Confirm the application was submitted successfully
		
		Important notes:
		- If login is required, try to find a guest application option first
		- Use accurate information from my CV for all form fields
		- If the job requires specific qualifications not in my CV, mention this in the final result
		- Always be honest and accurate in your responses
		- My full CV information is provided above in the CV INFORMATION section
		"""
		
		# Prepare CV file path for the agent
		cv_files = [str(self.cv_path)] if self.cv_path.exists() else []
		
		agent = Agent(
			task=task_description,
			llm=self.llm,
			browser_session=BrowserSession(browser_profile=self.browser_profile),
			controller=self.controller,
			use_vision=True,
			max_actions_per_step=15,
			available_file_paths=cv_files
		)
		
		try:
			history = await agent.run(max_steps=30)
			
			# Create application record
			application = JobApplication(
				job_id=job_data.id,
				job_title=job_data.title,
				company=job_data.company_name,
				location=location_info,
				job_url=job_data.url,
				notes=str(history.final_result()) if history.final_result() else "Application attempted"
			)
			
			self.applications.append(application)
			self.logger.info(f"Application completed: {application.job_title} at {application.company}")
			
			return application
			
		except Exception as e:
			self.logger.error(f"Failed to apply to job {job_data.url}: {e}")
			return None
	
	async def run_job_applications(self) -> list[JobApplication]:
		"""Run the complete job application process using jobs from JSON file"""
		self.logger.info("Starting job application process")
		
		if not self.cv_path.exists():
			self.logger.error(f"CV file not found: {self.cv_path}")
			return []
		
		if not self.available_jobs:
			self.logger.error("No jobs available to apply to")
			return []
		
		# Filter jobs based on config criteria
		filtered_jobs = self.filter_jobs(self.config.max_applications)
		
		self.logger.info(f"Found {len(filtered_jobs)} matching jobs to apply to")
		
		# Apply to each job
		for job in filtered_jobs:
			await self.apply_to_job(job)
			
			# Add delay between applications to be respectful
			await asyncio.sleep(5)
		
		# Save application log
		self._save_application_log()
		
		return self.applications
	
	def _save_application_log(self):
		"""Save application log to JSON file"""
		log_file = self.logs_dir / f"applications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
		
		with open(log_file, 'w') as f:
			json.dump(
				[app.model_dump() for app in self.applications],
				f,
				indent=2
			)
		
		self.logger.info(f"Application log saved to {log_file}")
	
	def get_application_summary(self) -> dict[str, Any]:
		"""Get summary of all applications"""
		return {
			"total_applications": len(self.applications),
			"applications": [app.model_dump() for app in self.applications],
			"config": self.config.model_dump()
		}


async def main():
	"""Main entry point for the job agent"""
	import os
	
	# Configuration
	config = JobSearchConfig(
		keywords=["python", "software engineer", "developer", "typescript", "react"],
		locations=["remote", "australia", "united states", "san francisco"],
		max_applications=5,
		target_sites=["workable.com", "ashbyhq.com"]  # Based on URLs in the JSON
	)
	
	# Initialize job agent
	agent = JobAgent(
		cv_path="cv.pdf",
		deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
		headless=False,  # Set to True for headless mode
		config=config,
		jobs_file="jobsfetch.json"
	)
	
	# Run job applications
	applications = await agent.run_job_applications()
	
	# Print summary
	summary = agent.get_application_summary()
	print("\n=== Job Application Summary ===")
	print(f"Total applications: {summary['total_applications']}")
	
	for app in applications:
		print(f"\n- {app.job_title} at {app.company}")
		print(f"  Location: {app.location}")
		print(f"  Applied: {app.application_date}")
		print(f"  Status: {app.status}")
		print(f"  Job ID: {app.job_id}")


if __name__ == "__main__":
	asyncio.run(main())