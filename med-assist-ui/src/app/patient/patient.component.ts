import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { NgxMicRecorderModule } from 'ngx-mic-recorder';


@Component({
  selector: 'app-patient',
  standalone: true,
  imports: [ReactiveFormsModule, CommonModule, NgxMicRecorderModule],
  templateUrl: './patient.component.html',
  styleUrl: './patient.component.css'
})
export class PatientComponent {
  protected patient: Patient = new Patient();
  patientForm: FormGroup;

  constructor(private fb: FormBuilder) {
    // Initialize form with FormBuilder
    this.patientForm = this.fb.group({
      name: ['', Validators.required],
      diagnosis: ['', Validators.required]
    });
  }

  saveAsBlob(blob : any) {
    console.log(blob);
  }
  afterStop(blob : any) {
    console.log(blob);
  }
}

export class Patient {
  name: string | undefined;
  diagnosis: string | undefined;
}