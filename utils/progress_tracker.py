import streamlit as st
from datetime import datetime

class ProgressTracker:
    def __init__(self):
        if 'completed_classes' not in st.session_state:
            st.session_state.completed_classes = set()
        if 'class_scores' not in st.session_state:
            st.session_state.class_scores = {}
        if 'start_time' not in st.session_state:
            st.session_state.start_time = datetime.now()
    
    def mark_class_completed(self, class_name):
        """Mark a class as completed"""
        st.session_state.completed_classes.add(class_name)
    
    def get_completed_classes(self):
        """Get list of completed classes"""
        return st.session_state.completed_classes
    
    def get_progress(self):
        """Calculate overall progress percentage"""
        total_classes = 10
        completed = len(st.session_state.completed_classes)
        return (completed / total_classes) * 100
    
    def set_class_score(self, class_name, score):
        """Set score for a specific class"""
        st.session_state.class_scores[class_name] = score
    
    def get_class_score(self, class_name):
        """Get score for a specific class"""
        return st.session_state.class_scores.get(class_name, 0)
    
    def reset_progress(self):
        """Reset all progress"""
        st.session_state.completed_classes = set()
        st.session_state.class_scores = {}
        st.session_state.start_time = datetime.now()
