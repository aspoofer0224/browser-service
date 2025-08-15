#!/usr/bin/env python3
"""
Job Agent - Automated job application using browser-use library
Applies for jobs using CV information from cv.pdf file
"""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import pdfplumber
import PyPDF2
from pydantic import BaseModel, Field

from browser_use import Agent, BrowserProfile, Controller
from browser_use.agent.views import ActionResult
from browser_use.browser.session import BrowserSession
from browser_use.llm import ChatOpenAI


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
		openai_api_key: str | None = None,
		twocaptcha_api_key: str | None = None,
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
		self.twocaptcha_api_key = twocaptcha_api_key or os.getenv("TWOCAPTCHA_API_KEY")
		
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
		self.llm = ChatOpenAI(
			model="gpt-4o",
			api_key=openai_api_key
		)
		
		# Browser profile with job site access and stealth mode
		self.browser_profile = BrowserProfile(
			headless=headless,
			user_data_dir=str(self.logs_dir / "browser_profile"),
			window_size={'width': 1920, 'height': 1080},
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
		
		@self.controller.action("Solve reCAPTCHA on current page")
		async def solve_recaptcha(browser_session: BrowserSession) -> ActionResult:
			"""Solve reCAPTCHA using 2Captcha service"""
			if not self.twocaptcha_api_key:
				self.logger.error("2Captcha API key not provided")
				return ActionResult(
					extracted_content="2Captcha API key not provided",
					success=False
				)
			
			try:
				page = await browser_session.get_current_page()
				# Check if reCAPTCHA is present
				recaptcha = await page.locator('div.g-recaptcha').first
				if not recaptcha:
					return ActionResult(
						extracted_content="No reCAPTCHA found on page",
						success=False
					)
				
				site_key = await recaptcha.get_attribute("data-sitekey")
				if not site_key:
					return ActionResult(
						extracted_content="Could not find reCAPTCHA site key",
						success=False
					)
				
				# Send CAPTCHA to 2Captcha
				async with aiohttp.ClientSession() as session:
					response = await session.post(
						"http://2captcha.com/in.php",
						data={
							"key": self.twocaptcha_api_key,
							"method": "userrecaptcha",
							"googlekey": site_key,
							"pageurl": page.url,
							"json": 1
						}
					)
					result = await response.json()
					if result["status"] != 1:
						return ActionResult(
							extracted_content=f"2Captcha request failed: {result['request']}",
							success=False
						)
					
					captcha_id = result["request"]
					
					# Poll for CAPTCHA solution
					for _ in range(12):  # Try for ~60 seconds
						await asyncio.sleep(5)
						result = await session.get(
							f"http://2captcha.com/res.php?key={self.twocaptcha_api_key}&action=get&id={captcha_id}&json=1"
						)
						response_data = await result.json()
						if response_data["status"] == 1:
							solution = response_data["request"]
							# Inject solution into page
							await page.evaluate(
								f"""document.getElementById('g-recaptcha-response').value = '{solution}';"""
							)
							self.logger.info("reCAPTCHA solved successfully")
							return ActionResult(
								extracted_content="reCAPTCHA solved successfully",
								success=True
							)
						elif response_data["request"] != "CAPCHA_NOT_READY":
							return ActionResult(
								extracted_content=f"2Captcha solving failed: {response_data['request']}",
								success=False
							)
					
					return ActionResult(
						extracted_content="2Captcha solving timed out",
						success=False
					)
			except Exception as e:
				self.logger.error(f"Failed to solve reCAPTCHA: {e}")
				return ActionResult(
					extracted_content=f"reCAPTCHA solving failed: {str(e)}",
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
		2. Wait for the page to fully load and handle any security checks
		3. If a reCAPTCHA is present, use the 'Solve reCAPTCHA on current page' action to solve it
		4. Look for an "Apply" or "Apply Now" button and click it
		5. Fill out the application form using information from my CV above
		6. Upload the CV file when prompted (file path: {self.cv_path})
		7. Answer any screening questions based on my skills and experience from the CV
		8. For salary expectations, be conservative and research-based
		9. Submit the application
		10. Confirm the application was submitted successfully
		
		CAPTCHA & SECURITY HANDLING:
		- If a reCAPTCHA appears, use the 'Solve reCAPTCHA on current page' action with 2Captcha
		- For complex captchas (hCaptcha, Amazon-style), try to solve them but if unsuccessful, report the issue
		- If Cloudflare or other anti-bot protection appears, wait for it to complete automatically
		- If faced with "Are you human?" checks, click the verification box and wait
		- If captcha solving fails repeatedly, try refreshing the page once and retry
		
		FORM SELECTION & INTERACTION GUIDANCE:
		- For dropdown menus, click to expand first, then select the appropriate option
		- For radio buttons and checkboxes, click directly on the option text or the input element
		- For date pickers, look for calendar icons or click directly on date fields
		- For multi-select fields, hold Ctrl/Cmd while clicking multiple options if needed
		- If form fields are not responding, try clicking slightly outside the field first, then inside
		- For file uploads, ensure the file picker dialog opens before selecting the CV file
		- If an element seems unclickable, try scrolling it into view first
		- For cascading dropdowns (country->state->city), select in order and wait for each to populate
		
		ERROR HANDLING & FALLBACK STRATEGIES:
		- If login is required, try to find a guest application option first
		- If the page redirects to a login wall, look for "Apply without account" or similar options
		- If form submission fails, check for validation errors and fix them
		- If the site seems to detect automation, wait 2-3 seconds between actions
		- If elements are hard to find, try using different selectors or text content
		- If the application process seems blocked, try going back and starting over
		- Document any persistent issues in your final response
		
		IMPORTANT NOTES:
		- Use accurate information from my CV for all form fields
		- If the job requires specific qualifications not in my CV, mention this in the final result
		- Always be honest and accurate in your responses
		- Take your time with each step - rushing often causes detection
		- My full CV information is provided above in the CV INFORMATION section
		- If you encounter repeated failures, provide a detailed error report
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
			
			# Track the application in records.json
			self.track_job_application(
				job_data=job_data,
				application_status="applied",
				notes=str(history.final_result()) if history.final_result() else "Application attempted"
			)
			
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
		
		# Check for 2Captcha API key
		if not self.twocaptcha_api_key:
			self.logger.warning("No 2Captcha API key provided; reCAPTCHA solving may fail")
		
		# Filter jobs based on config criteria
		filtered_jobs = self.filter_jobs(self.config.max_applications)
		
		self.logger.info(f"Found {len(filtered_jobs)} matching jobs to apply to")
		
		# If max_applications is 0, just return without applying
		if self.config.max_applications == 0:
			self.logger.info("Max applications set to 0, skipping job applications")
			return []
		
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
	
	def track_job_application(self, job_data: JobData, application_status: str = "applied", notes: str = "") -> None:
		"""Track job application in records.json file"""
		records_file = Path("records.json")
		
		# Create the tracking record
		record = {
			"job_id": job_data.id,
			"job_title": job_data.title,
			"company": job_data.company_name,
			"job_url": job_data.url,
			"salary": job_data.salary,
			"location": {
				"cities": job_data.cities,
				"countries": job_data.countries,
				"is_remote": job_data.is_remote
			},
			"seniority": job_data.seniority,
			"hours": job_data.hours,
			"description": job_data.description,
			"application_date": datetime.now().isoformat(),
			"status": application_status,
			"notes": notes,
			"created_at": job_data.created_at
		}
		
		# Load existing records or create new list
		try:
			if records_file.exists():
				with open(records_file, 'r') as f:
					records = json.load(f)
			else:
				records = []
		except (json.JSONDecodeError, FileNotFoundError):
			records = []
		
		# Check if this job is already tracked
		job_exists = any(r.get("job_id") == job_data.id for r in records)
		
		if not job_exists:
			records.append(record)
			
			# Save updated records
			with open(records_file, 'w') as f:
				json.dump(records, f, indent=2)
			
			self.logger.info(f"Job application tracked in records.json: {job_data.title} at {job_data.company_name}")
		else:
			self.logger.info(f"Job already tracked in records.json: {job_data.title} at {job_data.company_name}")
	
	def get_application_summary(self) -> dict[str, Any]:
		"""Get summary of all applications"""
		return {
			"total_applications": len(self.applications),
			"applications": [app.model_dump() for app in self.applications],
			"config": self.config.model_dump()
		}


async def main():
	"""Main entry point for the job agent"""
	import argparse
	import os
	import sys
	
	# Set up argument parser
	parser = argparse.ArgumentParser(description="Automated job application using browser automation")
	parser.add_argument("--cv-path", default="cv.pdf", help="Path to CV PDF file")
	parser.add_argument("--headless", type=str, default="false", help="Run browser in headless mode (true/false)")
	parser.add_argument("--jobs-file", default="jobsfetch.json", help="Path to jobs JSON file")
	parser.add_argument("--config", help="JSON string with job search configuration")
	parser.add_argument("--execution-id", help="Unique execution ID for tracking")
	
	args = parser.parse_args()
	
	# Convert headless string to boolean
	headless_mode = args.headless.lower() in ('true', '1', 'yes', 'on')
	print(f"Headless mode: {headless_mode} (from argument: {args.headless})", file=sys.stderr)
	
	# Get OpenAI API key from environment
	openai_api_key = os.getenv("OPENAI_API_KEY")
	if not openai_api_key:
		print("Error: OPENAI_API_KEY environment variable is required", file=sys.stderr)
		sys.exit(1)
	
	# Get 2Captcha API key from environment
	twocaptcha_api_key = os.getenv("TWOCAPTCHA_API_KEY")
	if not twocaptcha_api_key:
		print("Warning: TWOCAPTCHA_API_KEY environment variable not set; reCAPTCHA solving may fail", file=sys.stderr)
	
	# Parse configuration from JSON if provided
	if args.config:
		try:
			config_dict = json.loads(args.config)
			config = JobSearchConfig(**config_dict)
		except Exception as e:
			print(f"Error parsing config JSON: {e}", file=sys.stderr)
			sys.exit(1)
	else:
		# Default configuration
		config = JobSearchConfig(
			keywords=["python", "software engineer", "developer", "typescript", "react"],
			locations=["remote", "australia", "united states", "san francisco"],
			max_applications=5,
			target_sites=["workable.com", "ashbyhq.com"]
		)
	
	try:
		# Initialize job agent
		agent = JobAgent(
			cv_path=args.cv_path,
			openai_api_key=openai_api_key,
			twocaptcha_api_key=twocaptcha_api_key,
			headless=headless_mode,
			config=config,
			jobs_file=args.jobs_file
		)
		
		# Run job applications
		applications = await agent.run_job_applications()
		
		# Get summary
		summary = agent.get_application_summary()
		
		# Print human-readable summary to stderr (so it doesn't interfere with JSON output)
		print("\n=== Job Application Summary ===", file=sys.stderr)
		print(f"Total applications: {summary['total_applications']}", file=sys.stderr)
		print(f"Execution ID: {args.execution_id}", file=sys.stderr)
		
		for app in applications:
			print(f"\n- {app.job_title} at {app.company}", file=sys.stderr)
			print(f"  Location: {app.location}", file=sys.stderr)
			print(f"  Applied: {app.application_date}", file=sys.stderr)
			print(f"  Status: {app.status}", file=sys.stderr)
			print(f"  Job ID: {app.job_id}", file=sys.stderr)
		
		# Output structured JSON result to stdout for the backend to parse
		result = {
			"success": True,
			"total_applications": summary['total_applications'],
			"applications": [app.model_dump() for app in applications],
			"message": f"Successfully processed {len(applications)} job applications",
			"execution_id": args.execution_id,
			"config": config.model_dump()
		}
		
		print(f"RESULT:{json.dumps(result)}")
		
	except Exception as e:
		# Print error to stderr
		print(f"Error during job automation: {e}", file=sys.stderr)
		
		# Output error result to stdout
		error_result = {
			"success": False,
			"total_applications": 0,
			"applications": [],
			"message": f"Job automation failed: {str(e)}",
			"execution_id": args.execution_id,
			"error": str(e)
		}
		
		print(f"RESULT:{json.dumps(error_result)}")
		sys.exit(1)


if __name__ == "__main__":
	asyncio.run(main())