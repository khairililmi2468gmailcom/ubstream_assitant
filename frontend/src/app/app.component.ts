// frontend/src/app/app.component.ts

import { Component, OnInit } from '@angular/core';
import { ApiService } from './api.service';
import { CommonModule } from '@angular/common'; // Required for ngClass, ngIf, ngFor
import { FormsModule } from '@angular/forms';   // Required for ngModel

interface AnswerHistoryItem {
  question: string;
  answer: string;
  inferenceTime: string;
  timestamp: number; // Unix timestamp for sorting by time
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  standalone: true,
  imports: [
    CommonModule,
    FormsModule
  ]
})
export class AppComponent implements OnInit {
  title = 'Ubestream RAG Assistant';
  question: string = '';
  answer: string = '';
  inferenceTime: string = '';
  isLoading: boolean = false;
  apiStatus: string = 'Checking...';
  errorMessage: string = '';

  // History related properties
  answerHistory: AnswerHistoryItem[] = [];
  filteredHistory: AnswerHistoryItem[] = []; // History after applying sort/filter
  currentSort: 'time_desc' | 'time_asc' | 'alpha_asc' | 'alpha_desc' = 'time_desc';
  historyLimit: number = 10; // Number of history items to show initially, before scroll

  constructor(private apiService: ApiService) {}

  ngOnInit(): void {
    this.checkApiStatus();
    this.loadHistoryFromLocalStorage();
    this.applySortingAndFiltering(); // Apply initial sorting/filtering
  }

  /**
   * Loads answer history from local storage.
   */
  private loadHistoryFromLocalStorage(): void {
    try {
      const historyJson = localStorage.getItem('ubestream_rag_history');
      if (historyJson) {
        this.answerHistory = JSON.parse(historyJson);
        // Ensure timestamps are numbers
        this.answerHistory.forEach(item => item.timestamp = Number(item.timestamp));
        console.log('History loaded from local storage:', this.answerHistory.length, 'items');
      }
    } catch (e) {
      console.error('Failed to load history from local storage', e);
      this.answerHistory = []; // Reset if corrupted
    }
  }

  /**
   * Saves current answer history to local storage.
   */
  private saveHistoryToLocalStorage(): void {
    try {
      localStorage.setItem('ubestream_rag_history', JSON.stringify(this.answerHistory));
      console.log('History saved to local storage.');
    } catch (e) {
      console.error('Failed to save history to local storage', e);
    }
  }

  /**
   * Clears all answer history from memory and local storage.
   */
  clearHistory(): void {
    if (confirm('Are you sure you want to clear all history?')) {
      this.answerHistory = [];
      this.filteredHistory = [];
      localStorage.removeItem('ubestream_rag_history');
      console.log('History cleared.');
    }
  }

  /**
   * Applies current sorting and filtering to the answer history.
   */
  applySortingAndFiltering(): void {
    let sorted = [...this.answerHistory]; // Create a shallow copy to sort

    switch (this.currentSort) {
      case 'time_desc':
        sorted.sort((a, b) => b.timestamp - a.timestamp); // Newest first
        break;
      case 'time_asc':
        sorted.sort((a, b) => a.timestamp - b.timestamp); // Oldest first
        break;
      case 'alpha_asc':
        sorted.sort((a, b) => a.question.localeCompare(b.question)); // A-Z by question
        break;
      case 'alpha_desc':
        sorted.sort((a, b) => b.question.localeCompare(a.question)); // Z-A by question
        break;
    }

    this.filteredHistory = sorted;
  }

  /**
   * Checks the status of the backend API.
   */
  checkApiStatus(): void {
    this.apiStatus = 'Checking...';
    this.apiService.checkStatus().subscribe({
      next: (response) => {
        this.apiStatus = response.status === 'ready' ? 'Ready' : 'Not Ready';
        if (response.status !== 'ready') {
          this.errorMessage = response.message || 'API is not ready.';
        } else {
          this.errorMessage = '';
        }
      },
      error: (err) => {
        this.apiStatus = 'Error';
        this.errorMessage = 'Could not connect to backend API. Please ensure FastAPI server is running on http://localhost:8006.';
        console.error('API Status Check Error:', err);
      }
    });
  }

  /**
   * Sends the user's question to the backend API.
   */
  sendQuestion(): void {
    if (!this.question.trim()) {
      this.errorMessage = 'Please enter a question.';
      return;
    }

    this.isLoading = true;
    this.answer = ''; // Clear previous answer in display area
    this.inferenceTime = '';
    this.errorMessage = '';

    const currentQuestion = this.question; // Store question before clearing input

    this.apiService.askQuestion(currentQuestion).subscribe({
      next: (response) => {
        this.answer = response.answer || 'No answer received.';
        this.inferenceTime = response.inference_time || 'N/A';
        this.isLoading = false;

        // Add to history
        const newHistoryItem: AnswerHistoryItem = {
          question: currentQuestion,
          answer: this.answer,
          inferenceTime: this.inferenceTime,
          timestamp: Date.now() // Current timestamp
        };
        this.answerHistory.push(newHistoryItem); // Add to the end, then sort
        this.saveHistoryToLocalStorage();
        this.applySortingAndFiltering(); // Re-sort history after adding new item

        this.question = ''; // Clear input field after successful submission
      },
      error: (err) => {
        this.errorMessage = 'Failed to get answer. Check console for details. Ensure Ollama server is running and models are pulled.';
        this.isLoading = false;
        console.error('Ask Question Error:', err);
      }
    });
  }
}
