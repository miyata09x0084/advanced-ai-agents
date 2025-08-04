#!/usr/bin/env python3
"""
Translator Agent - Literary translation with evaluator-optimizer workflow

This script implements a literary translation system where one LLM generates translations
and another evaluates and provides feedback for improvement in an iterative loop. It behaves like a sequential agent in a loop. 
"""

import os
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import utility functions for structured outputs
from utils import *

# Model configurations
GENERATOR_MODEL = "gpt-4o"  # For generating translations
EVALUATOR_MODEL = "o3-mini"  # For evaluating translations

# Consolidated Pydantic models for structured outputs
class Translation(BaseModel):
    """Model for translation output."""
    translation: str = Field(..., description="The translated text")


class TranslationFeedback(BaseModel):
    """Model for translation feedback."""
    semantic_accuracy: float = Field(..., description="Score for semantic accuracy (1-10)")
    stylistic_authenticity: float = Field(..., description="Score for stylistic authenticity (1-10)")
    cultural_adaptation: float = Field(..., description="Score for cultural adaptation (1-10)")
    readability: float = Field(..., description="Score for readability (1-10)")
    literary_preservation: float = Field(..., description="Score for literary device preservation (1-10)")
    overall_score: float = Field(..., description="Overall score (1-10)")
    detailed_feedback: str = Field(..., description="Detailed critique and suggestions for improvement")
    
    @property
    def average_score(self) -> float:
        """Calculate the average score across all criteria."""
        return self.overall_score
    
    def dict(self) -> Dict[str, Any]:
        """Override the default dict method to ensure compatibility."""
        base_dict = super().model_dump()
        return base_dict


class TranslationIteration(BaseModel):
    """Model for storing a single iteration of the translation process."""
    iteration_number: int
    translation: str
    feedback: Optional[TranslationFeedback] = None
    generation_time: float = 0.0
    evaluation_time: float = 0.0


class TranslatorGenerator:
    """Responsible for generating literary translations."""
    
    def __init__(self, model: str = GENERATOR_MODEL):
        """Initialize the translator generator."""
        self.model = model
    
    def translate(self, text: str, source_language: str, target_language: str, 
                  previous_feedback: Optional[str] = None) -> Tuple[str, float]:
        """
        Translate text from source language to target language.
        
        Args:
            text: The text to translate
            source_language: The source language
            target_language: The target language
            previous_feedback: Optional feedback from previous iteration
            
        Returns:
            Tuple containing the translation and generation time
        """
        start_time = time.time()
        
        system_message = """
        You are an expert literary translator with deep knowledge of cultural nuances, 
        literary devices, and stylistic elements. Your task is to translate the provided 
        text while preserving its literary qualities, cultural context, and emotional impact.
        
        Focus on these aspects in your translation:
        1. Maintain the author's voice and style
        2. Preserve literary devices (metaphors, similes, etc.)
        3. Adapt cultural references appropriately
        4. Ensure idiomatic expressions and wordplay are translated effectively
        5. Preserve the rhythm and flow of the original text
        
        Provide only the translated text without explanations or commentary.
        """
        
        user_message = f"Translate the following text from {source_language} to {target_language}:\n\n{text}"
        
        # If there's previous feedback, include it
        if previous_feedback:
            user_message += f"\n\nBased on the following feedback, please improve your translation:\n{previous_feedback}"
        
        try:
            # Use structured output approach with simplified model
            result = get_structured_output(
                query=user_message,
                system_message=system_message,
                response_schema=Translation,
                description="translation"
            )
            
            if result:
                translation = result.translation
            else:
                # If structured output fails, raise an error
                print("Error: Failed to generate structured translation output")
                return "Error: Structured output generation failed", time.time() - start_time
                
            generation_time = time.time() - start_time
            return translation, generation_time
        
        except Exception as e:
            print(f"Error generating translation: {e}")
            return f"Error: {str(e)}", time.time() - start_time


class TranslationEvaluator:
    """Responsible for evaluating translations and providing feedback."""
    
    def __init__(self, model: str = EVALUATOR_MODEL):
        """Initialize the translation evaluator."""
        self.model = model
    
    def evaluate(self, original_text: str, translation: str, 
                 source_language: str, target_language: str) -> Tuple[TranslationFeedback, float]:
        """
        Evaluate a translation and provide detailed feedback.
        
        Args:
            original_text: The original text
            translation: The translated text to evaluate
            source_language: The source language
            target_language: The target language
            
        Returns:
            Tuple containing TranslationFeedback and evaluation time
        """
        start_time = time.time()
        
        system_message = """
        You are an expert literary critic and translator with deep knowledge of both the source and target languages. 
        Your task is to evaluate a literary translation based on several key criteria and provide detailed, 
        constructive feedback for improvement.
        
        Evaluate the translation on these criteria (score from 1-10):
        1. Semantic accuracy: How accurately the meaning is preserved
        2. Stylistic authenticity: How well the author's style is maintained
        3. Cultural adaptation: How appropriately cultural elements are handled
        4. Readability and flow: How natural and engaging the translation reads
        5. Literary device preservation: How well literary devices are translated
        
        Provide an overall score (1-10) and detailed feedback with specific suggestions for improvement.
        """
        
        user_message = f"""
        Please evaluate this literary translation:
        
        Original text ({source_language}):
        {original_text}
        
        Translation ({target_language}):
        {translation}
        
        Provide scores and detailed feedback for improvement.
        """
        
        try:
            # Use structured output approach with simplified model
            result = get_structured_output(
                query=user_message,
                system_message=system_message,
                response_schema=TranslationFeedback,
                description="translation feedback"
            )
            
            if not result:
                # If structured output fails, print error and return error message
                print("Error: Failed to generate structured translation feedback output")
                return "Error: Structured output generation failed", time.time() - start_time
            
            evaluation_time = time.time() - start_time
            return result, evaluation_time
            
        except Exception as e:
            print(f"Error evaluating translation: {e}")
            # Return default feedback with error message
            error_feedback = TranslationFeedback(
                semantic_accuracy=0,
                stylistic_authenticity=0,
                cultural_adaptation=0,
                readability=0,
                literary_preservation=0,
                overall_score=0,
                detailed_feedback=f"Error evaluating translation: {str(e)}"
            )
            return error_feedback, time.time() - start_time


class TranslationOptimizer:
    """Manages the evaluator-optimizer workflow for literary translation."""
    
    def __init__(self, 
                 generator_model: str = GENERATOR_MODEL, 
                 evaluator_model: str = EVALUATOR_MODEL,
                 max_iterations: int = 5,
                 quality_threshold: float = 8.5):
        """Initialize the translation optimizer."""
        self.generator = TranslatorGenerator(model=generator_model)
        self.evaluator = TranslationEvaluator(model=evaluator_model)
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.history: List[TranslationIteration] = []
    
    def optimize(self, text: str, source_language: str, target_language: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the translation optimization loop.
        
        Args:
            text: The text to translate
            source_language: The source language
            target_language: The target language
            verbose: Whether to print progress
            
        Returns:
            Dict with final translation, feedback, history, and metrics
        """
        if verbose:
            print(f"Starting translation optimization from {source_language} to {target_language}...")
            print(f"Maximum iterations: {self.max_iterations}")
            print(f"Quality threshold: {self.quality_threshold}")
            print("-" * 50)
            print("\nORIGINAL TEXT:")
            print("-" * 50)
            print(text)
            print("-" * 50)
        
        # Initial translation without feedback
        translation, generation_time = self.generator.translate(
            text, source_language, target_language
        )
        
        iteration = TranslationIteration(
            iteration_number=1,
            translation=translation,
            generation_time=generation_time
        )
        
        # Evaluate initial translation
        feedback, evaluation_time = self.evaluator.evaluate(
            text, translation, source_language, target_language
        )
        
        iteration.feedback = feedback
        iteration.evaluation_time = evaluation_time
        self.history.append(iteration)
        
        if verbose:
            self._print_iteration_results(iteration)
        
        current_score = feedback.average_score
        best_iteration = iteration
        
        # Optimization loop
        for i in range(2, self.max_iterations + 1):
            # Check if we've reached the quality threshold
            if current_score >= self.quality_threshold:
                if verbose:
                    print(f"\nQuality threshold reached! Score: {current_score}")
                break
            
            # Generate improved translation using previous feedback
            translation, generation_time = self.generator.translate(
                text, source_language, target_language, previous_feedback=feedback.detailed_feedback
            )
            
            iteration = TranslationIteration(
                iteration_number=i,
                translation=translation,
                generation_time=generation_time
            )
            
            # Evaluate the new translation
            feedback, evaluation_time = self.evaluator.evaluate(
                text, translation, source_language, target_language
            )
            
            iteration.feedback = feedback
            iteration.evaluation_time = evaluation_time
            self.history.append(iteration)
            
            if verbose:
                self._print_iteration_results(iteration)
            
            # Update best iteration if this one is better
            if feedback.average_score > current_score:
                current_score = feedback.average_score
                best_iteration = iteration
        
        # Prepare final results
        final_results = {
            "original_text": text,
            "source_language": source_language,
            "target_language": target_language,
            "final_translation": best_iteration.translation,
            "final_feedback": best_iteration.feedback.model_dump() if best_iteration.feedback else None,
            "iterations_completed": len(self.history),
            "best_iteration": best_iteration.iteration_number,
            "final_score": current_score,
            "time_metrics": {
                "total_generation_time": sum(iter.generation_time for iter in self.history),
                "total_evaluation_time": sum(iter.evaluation_time for iter in self.history),
                "total_time": sum(iter.generation_time + iter.evaluation_time for iter in self.history)
            },
            "history": [
                {
                    "iteration": iter.iteration_number,
                    "score": iter.feedback.average_score if iter.feedback else 0,
                    "translation_excerpt": iter.translation[:100] + "..." if len(iter.translation) > 100 else iter.translation
                }
                for iter in self.history
            ]
        }
        
        if verbose:
            self._print_final_results(final_results)
        
        return final_results
    
    def _print_iteration_results(self, iteration: TranslationIteration) -> None:
        """Print the results of an iteration."""
        feedback = iteration.feedback
        if not feedback:
            return
        
        print(f"\nIteration {iteration.iteration_number} Results:")
        print(f"Semantic Accuracy: {feedback.semantic_accuracy:.1f}/10")
        print(f"Stylistic Authenticity: {feedback.stylistic_authenticity:.1f}/10")
        print(f"Cultural Adaptation: {feedback.cultural_adaptation:.1f}/10")
        print(f"Readability: {feedback.readability:.1f}/10")
        print(f"Literary Preservation: {feedback.literary_preservation:.1f}/10")
        print(f"Overall Score: {feedback.overall_score:.1f}/10")
        print("\nTranslation Sample:")
        # Print first 100 characters of translation as a sample
        sample = iteration.translation[:100] + "..." if len(iteration.translation) > 100 else iteration.translation
        print(sample)
        print("\nFeedback Summary:")
        # Split feedback into lines and print first 5 lines
        feedback_lines = feedback.detailed_feedback.split("\n")
        print("\n".join(feedback_lines[:5]))
        if len(feedback_lines) > 5:
            print("...")
        print("-" * 50)
    
    def _print_final_results(self, results: Dict[str, Any]) -> None:
        """Print the final results of the optimization."""
        print("\n" + "=" * 60)
        print("FINAL TRANSLATION RESULTS")
        print("=" * 60)
        print(f"Source Language: {results['source_language']}")
        print(f"Target Language: {results['target_language']}")
        print(f"Iterations Completed: {results['iterations_completed']}/{self.max_iterations}")
        print(f"Best Iteration: {results['best_iteration']}")
        print(f"Final Score: {results['final_score']:.2f}/10")
        print(f"Total Time: {results['time_metrics']['total_time']:.2f} seconds")
        
        print("\nScore Progression:")
        for item in results['history']:
            print(f"  Iteration {item['iteration']}: {item['score']:.2f}/10")
        
        print("\nOriginal Text:")
        print("-" * 60)
        print(results['original_text'])
        print("-" * 60)
        
        print("\nFinal Translation:")
        print("-" * 60)
        print(results['final_translation'])
        print("-" * 60)
        
        # Display side by side if both texts are not too long
        if len(results['original_text']) + len(results['final_translation']) < 1000:
            print("\nSide by Side Comparison:")
            print("-" * 60)
            print(f"{'ORIGINAL':^40} | {'TRANSLATION':^40}")
            print("-" * 40 + "+" + "-" * 40)
            
            # Split text into lines and display side by side
            original_lines = results['original_text'].split("\n")
            translation_lines = results['final_translation'].split("\n")
            
            # Ensure both lists have the same length
            max_lines = max(len(original_lines), len(translation_lines))
            original_lines += [''] * (max_lines - len(original_lines))
            translation_lines += [''] * (max_lines - len(translation_lines))
            
            for o_line, t_line in zip(original_lines, translation_lines):
                # Truncate lines if they're too long
                o_display = (o_line[:37] + '...') if len(o_line) > 40 else o_line
                t_display = (t_line[:37] + '...') if len(t_line) > 40 else t_line
                print(f"{o_display:<40} | {t_display:<40}")
            
            print("-" * 60)
        
        if results['final_feedback']:
            print("\nFinal Feedback:")
            print(results['final_feedback']['detailed_feedback'])
        
        print("=" * 60)


def main():
    """Main function to run the literary translation optimization."""
    # Sample text for translation
    sample_texts = {
        "poetry": (
            "No te amo como si fueras rosa de sal, topacio\n"
            "o flecha de claveles que propagan el fuego:\n"
            "te amo como se aman ciertas cosas oscuras,\n"
            "secretamente, entre la sombra y el alma.",
            "Spanish",
            "English",
            "Poetry by Pablo Neruda (Sonnet XVII)"
        ),
        "prose": (
            "La memoria del corazón elimina los malos recuerdos y magnifica los buenos, "
            "y gracias a ese artificio, logramos sobrellevar el pasado.",
            "Spanish",
            "English",
            "Prose by Gabriel García Márquez"
        ),
        "idioms": (
            "En casa de herrero, cuchillo de palo. A otro perro con ese hueso. "
            "No hay que llorar sobre la leche derramada.",
            "Spanish",
            "English",
            "Spanish Idioms"
        )
    }
    
    print("Literary Translation Optimizer")
    print("============================")
    print("Select a sample text to translate:")
    print("1. Poetry (Pablo Neruda)")
    print("2. Prose (Gabriel García Márquez)")
    print("3. Spanish Idioms")
    print("4. Enter your own text")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == "1":
        text, source_lang, target_lang, description = sample_texts["poetry"]
        print(f"\nSelected: {description}")
    elif choice == "2":
        text, source_lang, target_lang, description = sample_texts["prose"]
        print(f"\nSelected: {description}")
    elif choice == "3":
        text, source_lang, target_lang, description = sample_texts["idioms"]
        print(f"\nSelected: {description}")
    elif choice == "4":
        print("Enter the text to translate (type 'END' on a new line when finished):")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        
        # Join with newlines and handle escaped sequences properly
        text = "\n".join(lines)
        text = text.replace("\\n", "\n")  # Convert literal \n to actual newlines
        
        source_lang = input("Enter the source language: ")
        target_lang = input("Enter the target language: ")
        description = "User provided text"
        
        # Display a preview of the text (first couple of lines)
        preview_lines = text.split("\n")[:3]
        preview = "\n".join(preview_lines)
        print(f"\nText to translate (preview): \n{preview}{'...' if len(preview_lines) > 2 else ''}")
    else:
        print("Invalid choice. Using poetry sample.")
        text, source_lang, target_lang, description = sample_texts["poetry"]
        print(f"\nSelected: {description}")
    
    # Configure optimizer
    max_iterations = int(input("Enter maximum iterations (2-10, default 3): ") or "3")
    max_iterations = max(2, min(10, max_iterations))  # Ensure between 2 and 10
    
    quality_threshold = float(input("Enter quality threshold (1-10, default 8.5): ") or "8.5")
    quality_threshold = max(1, min(10, quality_threshold))  # Ensure between 1 and 10
    
    # Initialize optimizer
    optimizer = TranslationOptimizer(
        max_iterations=max_iterations,
        quality_threshold=quality_threshold
    )
    
    # Run optimization
    results = optimizer.optimize(text, source_lang, target_lang)
    
    # Option to save results
    save_option = input("\nDo you want to save the results to a file? (y/n): ")
    if save_option.lower() == 'y':
        filename = input("Enter filename (default: translation_results.json): ") or "translation_results.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()
