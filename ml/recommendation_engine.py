"""
Recommendation Engine for Learning Disability Interventions
Provides personalized recommendations based on predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
import os

class RecommendationEngine:
    """
    Rule-based recommendation system for learning disabilities
    Provides intervention strategies, severity levels, and progress tracking
    """
    
    def __init__(self):
        self.intervention_database = self._load_intervention_database()
        self.severity_thresholds = {
            'mild': (0.40, 0.60),
            'moderate': (0.60, 0.80),
            'severe': (0.80, 1.00)
        }
    
    def _load_intervention_database(self) -> Dict:
        """
        Load intervention strategies database
        """
        return {
            'dyslexia': {
                'interventions': [
                    {
                        'name': 'Phonics-Based Reading Program',
                        'description': 'Structured phonics instruction focusing on sound-letter relationships',
                        'severity': ['mild', 'moderate', 'severe'],
                        'duration': '3-6 months',
                        'frequency': '4-5 times per week, 30-45 minutes',
                        'age_range': (6, 12),
                        'resources': [
                            'Wilson Reading System',
                            'Orton-Gillingham Approach',
                            'Reading mastery programs'
                        ]
                    },
                    {
                        'name': 'Multisensory Learning Techniques',
                        'description': 'Use visual, auditory, and kinesthetic methods simultaneously',
                        'severity': ['mild', 'moderate', 'severe'],
                        'duration': 'Ongoing',
                        'frequency': 'Daily practice, 20-30 minutes',
                        'age_range': (6, 12),
                        'resources': [
                            'Sand/rice trays for letter formation',
                            'Letter tiles and manipulatives',
                            'Audio books with text highlighting'
                        ]
                    },
                    {
                        'name': 'Assistive Technology',
                        'description': 'Text-to-speech and speech-to-text tools',
                        'severity': ['moderate', 'severe'],
                        'duration': 'Ongoing',
                        'frequency': 'As needed during reading/writing tasks',
                        'age_range': (7, 12),
                        'resources': [
                            'Learning Ally audiobooks',
                            'Read&Write literacy software',
                            'Voice typing tools'
                        ]
                    },
                    {
                        'name': 'One-on-One Tutoring',
                        'description': 'Specialized reading instruction with trained specialist',
                        'severity': ['moderate', 'severe'],
                        'duration': '6-12 months',
                        'frequency': '2-3 times per week, 45-60 minutes',
                        'age_range': (6, 12),
                        'resources': [
                            'Certified dyslexia specialist',
                            'Reading intervention programs',
                            'Progress monitoring tools'
                        ]
                    },
                    {
                        'name': 'Reading Comprehension Strategies',
                        'description': 'Explicit instruction in comprehension strategies',
                        'severity': ['mild', 'moderate'],
                        'duration': '3-6 months',
                        'frequency': '3-4 times per week, 30 minutes',
                        'age_range': (7, 12),
                        'resources': [
                            'Graphic organizers',
                            'Story maps',
                            'Question-answer relationships'
                        ]
                    }
                ],
                'classroom_accommodations': [
                    'Extended time on tests (1.5x to 2x)',
                    'Reduced reading assignments',
                    'Audio versions of textbooks',
                    'Use of text-to-speech software',
                    'Preferential seating near teacher',
                    'Frequent breaks during reading tasks',
                    'Highlighted or color-coded materials'
                ],
                'home_strategies': [
                    'Read aloud together daily (15-20 minutes)',
                    'Use audiobooks for enjoyment',
                    'Practice sight words with flashcards',
                    'Encourage writing through journaling',
                    'Limit screen time, increase reading time',
                    'Create a quiet, distraction-free study space'
                ]
            },
            'dysgraphia': {
                'interventions': [
                    {
                        'name': 'Occupational Therapy',
                        'description': 'Fine motor skill development and handwriting practice',
                        'severity': ['mild', 'moderate', 'severe'],
                        'duration': '3-6 months',
                        'frequency': '2-3 times per week, 30-45 minutes',
                        'age_range': (6, 12),
                        'resources': [
                            'Certified occupational therapist',
                            'Handwriting Without Tears program',
                            'Fine motor skill activities'
                        ]
                    },
                    {
                        'name': 'Handwriting Practice Programs',
                        'description': 'Structured handwriting instruction with proper letter formation',
                        'severity': ['mild', 'moderate', 'severe'],
                        'duration': '4-8 months',
                        'frequency': 'Daily practice, 15-20 minutes',
                        'age_range': (6, 12),
                        'resources': [
                            'Handwriting Without Tears',
                            'Size Matters Handwriting Program',
                            'Write Start program'
                        ]
                    },
                    {
                        'name': 'Assistive Technology for Writing',
                        'description': 'Speech-to-text and word prediction software',
                        'severity': ['moderate', 'severe'],
                        'duration': 'Ongoing',
                        'frequency': 'Daily use during writing tasks',
                        'age_range': (7, 12),
                        'resources': [
                            'Dragon NaturallySpeaking',
                            'Google Voice Typing',
                            'Co:Writer word prediction software'
                        ]
                    },
                    {
                        'name': 'Grip and Pencil Training',
                        'description': 'Proper pencil grip and pressure control',
                        'severity': ['mild', 'moderate'],
                        'duration': '2-4 months',
                        'frequency': 'Daily practice, 10-15 minutes',
                        'age_range': (6, 10),
                        'resources': [
                            'Pencil grips (triangular, cushioned)',
                            'Weighted pencils',
                            'Slant boards for writing surface'
                        ]
                    },
                    {
                        'name': 'Keyboarding Skills',
                        'description': 'Touch typing instruction as alternative to handwriting',
                        'severity': ['moderate', 'severe'],
                        'duration': '3-6 months',
                        'frequency': '3-4 times per week, 20-30 minutes',
                        'age_range': (7, 12),
                        'resources': [
                            'Typing.com',
                            'Typing Club',
                            'Dance Mat Typing'
                        ]
                    }
                ],
                'classroom_accommodations': [
                    'Allow use of computer/tablet for writing',
                    'Extended time on written assignments',
                    'Reduced writing requirements',
                    'Provide note-taking assistance',
                    'Accept oral responses instead of written',
                    'Use graph paper for math problems',
                    'Allow recording of lectures'
                ],
                'home_strategies': [
                    'Practice fine motor activities (playdough, beads)',
                    'Encourage drawing and coloring',
                    'Use lined paper with highlighted baseline',
                    'Break writing tasks into smaller chunks',
                    'Praise effort over neatness',
                    'Use multi-sensory writing activities (sand, shaving cream)'
                ]
            },
            'normal': {
                'interventions': [],
                'classroom_accommodations': [
                    'Continue regular monitoring',
                    'Encourage continued practice',
                    'Provide enrichment activities'
                ],
                'home_strategies': [
                    'Maintain regular reading routine',
                    'Encourage creative writing',
                    'Support academic growth'
                ]
            }
        }
    
    def determine_severity(self, prediction_proba: Dict[str, float], 
                          predicted_condition: str) -> str:
        """
        Determine severity level based on prediction confidence
        
        Args:
            prediction_proba: Dictionary of probabilities for each condition
            predicted_condition: The predicted disability
            
        Returns:
            Severity level: 'mild', 'moderate', or 'severe'
        """
        if predicted_condition == 'normal':
            return 'none'
        
        confidence = prediction_proba.get(predicted_condition, 0.0)
        
        for severity, (min_conf, max_conf) in self.severity_thresholds.items():
            if min_conf <= confidence < max_conf:
                return severity
        
        return 'severe' if confidence >= 0.80 else 'mild'
    
    def get_age_appropriate_interventions(self, interventions: List[Dict], 
                                         age: int) -> List[Dict]:
        """
        Filter interventions based on age appropriateness
        """
        age_appropriate = []
        for intervention in interventions:
            age_range = intervention.get('age_range', (6, 12))
            if age_range[0] <= age <= age_range[1]:
                age_appropriate.append(intervention)
        return age_appropriate
    
    def generate_recommendations(self, prediction: str, 
                                prediction_proba: Dict[str, float],
                                age: int = 8,
                                additional_context: Dict = None) -> Dict:
        """
        Generate comprehensive recommendations
        
        Args:
            prediction: Predicted condition (normal/dyslexia/dysgraphia)
            prediction_proba: Probability scores for each condition
            age: Student age
            additional_context: Additional student information
            
        Returns:
            Comprehensive recommendation dictionary
        """
        # Determine severity
        severity = self.determine_severity(prediction_proba, prediction)
        
        # Get interventions for this condition
        condition_data = self.intervention_database.get(prediction, {})
        all_interventions = condition_data.get('interventions', [])
        
        # Filter by severity and age
        severity_filtered = [
            i for i in all_interventions 
            if severity in i.get('severity', [])
        ]
        age_appropriate = self.get_age_appropriate_interventions(
            severity_filtered, age
        )
        
        # Prioritize interventions (top 3-5)
        priority_interventions = age_appropriate[:5]
        
        # Build recommendation response
        recommendations = {
            'prediction': prediction,
            'confidence': prediction_proba.get(prediction, 0.0),
            'severity_level': severity,
            'age': age,
            'primary_interventions': priority_interventions,
            'classroom_accommodations': condition_data.get('classroom_accommodations', []),
            'home_strategies': condition_data.get('home_strategies', []),
            'monitoring_plan': self._generate_monitoring_plan(prediction, severity),
            'next_steps': self._generate_next_steps(prediction, severity),
            'specialist_referral': self._should_refer_specialist(prediction, severity),
            'estimated_timeline': self._estimate_improvement_timeline(severity)
        }
        
        return recommendations
    
    def _generate_monitoring_plan(self, condition: str, severity: str) -> Dict:
        """
        Generate progress monitoring plan
        """
        monitoring_frequency = {
            'mild': 'Monthly assessments',
            'moderate': 'Bi-weekly assessments',
            'severe': 'Weekly assessments'
        }
        
        metrics = {
            'dyslexia': [
                'Reading speed (words per minute)',
                'Reading comprehension accuracy',
                'Phonics skills assessment',
                'Spelling accuracy',
                'Fluency scores'
            ],
            'dysgraphia': [
                'Handwriting legibility scores',
                'Letter formation quality',
                'Writing speed',
                'Fine motor skill assessments',
                'Written expression quality'
            ]
        }
        
        return {
            'frequency': monitoring_frequency.get(severity, 'Monthly'),
            'key_metrics': metrics.get(condition, []),
            'review_schedule': f'Every {3 if severity == "mild" else 6 if severity == "moderate" else 12} weeks',
            'reassessment_date': f'{3 if severity == "mild" else 2 if severity == "moderate" else 1} months'
        }
    
    def _generate_next_steps(self, condition: str, severity: str) -> List[str]:
        """
        Generate immediate next steps
        """
        steps = []
        
        if condition != 'normal':
            steps.append('Schedule meeting with school specialist/psychologist')
            steps.append('Inform parents about assessment results')
            
            if severity in ['moderate', 'severe']:
                steps.append('Request formal educational evaluation')
                steps.append('Consider IEP (Individualized Education Program) or 504 Plan')
                steps.append('Consult with occupational therapist or reading specialist')
            
            steps.append('Implement recommended classroom accommodations')
            steps.append('Begin intervention strategies immediately')
            steps.append(f'Schedule follow-up assessment in {2 if severity == "severe" else 3} months')
        else:
            steps.append('Continue regular monitoring')
            steps.append('Maintain current educational practices')
        
        return steps
    
    def _should_refer_specialist(self, condition: str, severity: str) -> Dict:
        """
        Determine if specialist referral is needed
        """
        if condition == 'normal':
            return {
                'required': False,
                'reason': 'No learning disability detected',
                'specialists': []
            }
        
        specialists = {
            'dyslexia': [
                'Educational Psychologist',
                'Reading Specialist/Literacy Coach',
                'Speech-Language Pathologist (if speech concerns)',
                'Certified Dyslexia Therapist'
            ],
            'dysgraphia': [
                'Occupational Therapist',
                'Educational Psychologist',
                'Special Education Teacher',
                'Physical Therapist (if gross motor concerns)'
            ]
        }
        
        urgency = {
            'mild': 'Recommended within 2-3 months',
            'moderate': 'Recommended within 1 month',
            'severe': 'Urgent - schedule within 2 weeks'
        }
        
        return {
            'required': True,
            'urgency': urgency.get(severity, 'Recommended'),
            'reason': f'{severity.capitalize()} {condition} detected',
            'specialists': specialists.get(condition, [])
        }
    
    def _estimate_improvement_timeline(self, severity: str) -> str:
        """
        Estimate timeline for improvement
        """
        timelines = {
            'mild': '3-6 months with consistent intervention',
            'moderate': '6-12 months with intensive intervention',
            'severe': '12-24 months with comprehensive intervention program'
        }
        return timelines.get(severity, 'Varies by individual')
    
    def generate_progress_report(self, student_id: str, 
                                 assessments: List[Dict]) -> Dict:
        """
        Generate progress tracking report from multiple assessments
        
        Args:
            student_id: Student identifier
            assessments: List of assessment results over time
            
        Returns:
            Progress report with trends and recommendations
        """
        if not assessments:
            return {'error': 'No assessments provided'}
        
        # Sort by date
        sorted_assessments = sorted(
            assessments, 
            key=lambda x: x.get('date', '')
        )
        
        # Calculate trends
        confidence_scores = [
            a.get('confidence', 0) for a in sorted_assessments
        ]
        
        # Determine improvement
        if len(confidence_scores) > 1:
            recent_avg = np.mean(confidence_scores[-3:])
            earlier_avg = np.mean(confidence_scores[:3] if len(confidence_scores) > 3 else confidence_scores)
            improvement = earlier_avg - recent_avg  # Lower confidence in disability = improvement
        else:
            improvement = 0
        
        trend = 'improving' if improvement > 0.05 else 'declining' if improvement < -0.05 else 'stable'
        
        return {
            'student_id': student_id,
            'total_assessments': len(assessments),
            'date_range': {
                'first': sorted_assessments[0].get('date', 'Unknown'),
                'last': sorted_assessments[-1].get('date', 'Unknown')
            },
            'trend': trend,
            'improvement_rate': float(improvement),
            'latest_status': sorted_assessments[-1],
            'recommendations': self._generate_progress_based_recommendations(trend, improvement)
        }
    
    def _generate_progress_based_recommendations(self, trend: str, 
                                                 improvement: float) -> List[str]:
        """
        Generate recommendations based on progress
        """
        recommendations = []
        
        if trend == 'improving':
            recommendations.append('Continue current intervention strategies')
            recommendations.append('Gradually increase task difficulty')
            recommendations.append('Consider reducing intensity of some interventions')
        elif trend == 'declining':
            recommendations.append('Review and adjust current intervention strategies')
            recommendations.append('Increase frequency of specialist consultations')
            recommendations.append('Consider additional assessments to identify barriers')
        else:  # stable
            recommendations.append('Maintain current intervention approaches')
            recommendations.append('Monitor for any changes in next 4-6 weeks')
            recommendations.append('Consider introducing new intervention techniques')
        
        return recommendations
    
    def export_recommendations(self, recommendations: Dict, 
                              format: str = 'json') -> str:
        """
        Export recommendations in various formats
        
        Args:
            recommendations: Recommendation dictionary
            format: 'json' or 'text'
            
        Returns:
            Formatted string
        """
        if format == 'json':
            return json.dumps(recommendations, indent=2)
        
        elif format == 'text':
            text = f"""
LEARNING DISABILITY ASSESSMENT REPORT
{'='*60}

PREDICTION: {recommendations['prediction'].upper()}
Confidence: {recommendations['confidence']*100:.1f}%
Severity Level: {recommendations['severity_level'].upper()}
Student Age: {recommendations['age']} years

PRIMARY INTERVENTIONS:
"""
            for i, intervention in enumerate(recommendations['primary_interventions'], 1):
                text += f"\n{i}. {intervention['name']}\n"
                text += f"   Description: {intervention['description']}\n"
                text += f"   Duration: {intervention['duration']}\n"
                text += f"   Frequency: {intervention['frequency']}\n"
            
            text += f"\n\nCLASSROOM ACCOMMODATIONS:\n"
            for acc in recommendations['classroom_accommodations']:
                text += f"• {acc}\n"
            
            text += f"\n\nHOME STRATEGIES:\n"
            for strategy in recommendations['home_strategies']:
                text += f"• {strategy}\n"
            
            text += f"\n\nNEXT STEPS:\n"
            for step in recommendations['next_steps']:
                text += f"• {step}\n"
            
            text += f"\n\nSPECIALIST REFERRAL:\n"
            ref = recommendations['specialist_referral']
            text += f"Required: {ref['required']}\n"
            if ref['required']:
                text += f"Urgency: {ref['urgency']}\n"
                text += f"Recommended Specialists:\n"
                for spec in ref['specialists']:
                    text += f"  • {spec}\n"
            
            return text
        
        return str(recommendations)


if __name__ == "__main__":
    # Test recommendation engine
    engine = RecommendationEngine()
    
    # Test case 1: Moderate dyslexia
    print("="*60)
    print("TEST CASE 1: Moderate Dyslexia")
    print("="*60)
    
    prediction_proba = {
        'normal': 0.10,
        'dyslexia': 0.75,
        'dysgraphia': 0.15
    }
    
    recommendations = engine.generate_recommendations(
        prediction='dyslexia',
        prediction_proba=prediction_proba,
        age=8
    )
    
    print(engine.export_recommendations(recommendations, format='text'))
    
    # Test case 2: Severe dysgraphia
    print("\n" + "="*60)
    print("TEST CASE 2: Severe Dysgraphia")
    print("="*60)
    
    prediction_proba = {
        'normal': 0.05,
        'dyslexia': 0.10,
        'dysgraphia': 0.85
    }
    
    recommendations = engine.generate_recommendations(
        prediction='dysgraphia',
        prediction_proba=prediction_proba,
        age=7
    )
    
    print(engine.export_recommendations(recommendations, format='text'))
    
    # Test case 3: Progress tracking
    print("\n" + "="*60)
    print("TEST CASE 3: Progress Tracking")
    print("="*60)
    
    assessments = [
        {'date': '2025-09-01', 'confidence': 0.85, 'condition': 'dyslexia'},
        {'date': '2025-10-01', 'confidence': 0.78, 'condition': 'dyslexia'},
        {'date': '2025-11-01', 'confidence': 0.65, 'condition': 'dyslexia'},
    ]
    
    progress = engine.generate_progress_report('STU001', assessments)
    print(json.dumps(progress, indent=2))
