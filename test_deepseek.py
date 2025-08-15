#!/usr/bin/env python3
"""
Simple test for ChatDeepSeek implementation
"""

import asyncio
import os
from deepseekjob import ChatDeepSeek
from browser_use.llm.messages import UserMessage

async def test_deepseek():
	"""Test basic ChatDeepSeek functionality"""
	
	# Check if API key is available
	api_key = os.getenv("DEEPSEEK_API_KEY")
	if not api_key:
		print("❌ DEEPSEEK_API_KEY environment variable not set")
		print("   Please set it with: export DEEPSEEK_API_KEY=your_api_key")
		return False
	
	try:
		# Initialize ChatDeepSeek
		llm = ChatDeepSeek(
			model='deepseek-chat',
			api_key=api_key,
			temperature=0.1
		)
		
		print(f"✅ ChatDeepSeek initialized successfully")
		print(f"   Provider: {llm.provider}")
		print(f"   Model: {llm.name}")
		print(f"   Base URL: {llm.base_url}")
		
		# Test basic message
		messages = [UserMessage(content="Hello! Please respond with 'DeepSeek is working correctly.'")]
		
		print("\n🔄 Testing basic message...")
		response = await llm.ainvoke(messages)
		
		print(f"✅ Response received:")
		print(f"   Content: {response.completion}")
		if response.usage:
			print(f"   Tokens - Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")
		
		return True
		
	except Exception as e:
		print(f"❌ Test failed: {e}")
		return False

if __name__ == "__main__":
	success = asyncio.run(test_deepseek())
	if success:
		print("\n🎉 ChatDeepSeek implementation is working correctly!")
	else:
		print("\n💥 ChatDeepSeek implementation needs debugging.")