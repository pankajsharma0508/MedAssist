import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { NgxMicRecorderModule } from 'ngx-mic-recorder';
import { lastValueFrom } from 'rxjs';
import { HttpClient, HttpClientModule } from '@angular/common/http';


@Component({
  selector: 'app-patient',
  standalone: true,
  imports: [FormsModule, CommonModule, NgxMicRecorderModule, HttpClientModule],
  templateUrl: './patient.component.html',
  styleUrl: './patient.component.css'
})
export class PatientComponent {
  protected patient: Patient = new Patient();
  apiUrl = 'http://localhost:8000';
  protected files: string | undefined;

  step = 1;

  constructor(private http: HttpClient) {
  }

  async saveAsBlob(blob: any) {
    const diagnosis = await lastValueFrom(this.convertSpeechToText(blob));
    this.patient.diagnosis = diagnosis;
    alert(diagnosis);
  }

  afterStop(blob: any) {
    console.log(blob);
  }

  public convertSpeechToText(file: File) {
    const formData: FormData = new FormData();
    formData.append('file', file, 'audio.mp3');
    return this.http.post<any>(`${this.apiUrl}/speech-to-text`, formData);
  }

  async getSeverity() {
    const severity = await lastValueFrom(this.http.get<any>(`${this.apiUrl}/patient-severity?symptoms=${encodeURIComponent(`${this.patient.symptoms}`)}`));
    this.patient.severity = severity;
  }

  async summarize() {
    const details = `${this.patient.name} report that ${this.patient.symptoms}. We considered him as a ${this.patient.severity}. On further diagnosis, found that, ${this.patient.diagnosis}`;
    const summary = await lastValueFrom(this.http.get<any>(`${this.apiUrl}/summarize?details=${encodeURIComponent(`${details}`)}`));
    this.patient.summary = summary;
  }

  async diagnose() {
    const summary = await lastValueFrom(this.http.get<any>(`${this.apiUrl}/predict-disease?symptoms=${encodeURIComponent(`${this.patient.symptoms}`)}`));
    this.patient.diagnosis = `${this.patient.diagnosis} \n ${summary}`;
  }

  async categoriesReports() {
    const files = this.files?.split(';');
    this.patient.reports = new Array<ImageWithCategory>();
    if (files) {
      files.forEach(async (imageUrl, index) => {
        const category = await lastValueFrom(this.http.get<any>(`${this.apiUrl}/image-category?imageUrl=${encodeURIComponent(`${imageUrl}`)}`));
        const image = new ImageWithCategory();
        image.category = category;
        image.name = `Report ${index}`;
        image.url = imageUrl;
        this.patient.reports.push(image);
      });
    }
  }
}
export class Patient {
  name: string | undefined;
  diagnosis: string | undefined;
  symptoms: string | undefined;
  severity: string | undefined
  summary: string | undefined
  reports: Array<ImageWithCategory> = []
}

export class ImageWithCategory {
  name: string | undefined;
  url: string | undefined;
  category: string | undefined;
}