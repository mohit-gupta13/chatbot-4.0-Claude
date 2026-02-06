"""
Data Service Module
Handles all data access operations from Excel files
NO LLM logic - pure data operations only
"""

import pandas as pd
import warnings
from typing import List, Dict, Optional, Any
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class DataService:
    """
    Data access layer for travel agency data.
    All database/file operations happen here.
    """
    
    # Status code mappings as requested
    STATUS_CODES = {
        'HK': 'Confirmed',
        'UC': 'On Request',
        'CL': 'Cancelled',
        'TKT': 'Ticketed',
        'RQ': 'On Request (Hotel)'
    }
    
    def __init__(self, excel_file: Optional[str] = None):
        if excel_file is None:
            try:
                from config import Config
                excel_file = Config.BOOKING_FILE
            except:
                excel_file = "Existing Booking.xlsx"
        
        self.excel_file = excel_file
        self._df = None
        
    def _load_data(self) -> pd.DataFrame:
        """
        Load data from Excel file with error handling.
        Uses caching to avoid repeated file reads.
        """
        if self._df is None:
            try:
                # Try multiple engines to handle different Excel formats
                try:
                    self._df = pd.read_excel(self.excel_file, engine='openpyxl')
                except Exception as e:
                    print(f"FAILED openpyxl: {e}. Trying xlrd...")
                    try:
                        self._df = pd.read_excel(self.excel_file, engine='xlrd')
                    except Exception as e2:
                        print(f"FAILED xlrd: {e2}. Trying default engine...")
                        self._df = pd.read_excel(self.excel_file)
                
                # Basic cleaning
                if not self._df.empty:
                    # Strip whitespace from column names
                    self._df.columns = self._df.columns.str.strip()
                    # Apply status code mapping
                    if 'Status' in self._df.columns:
                        self._df['Status_Label'] = self._df['Status'].astype(str).str.strip().map(self.STATUS_CODES).fillna(self._df['Status'])
                    else:
                        self._df['Status_Label'] = "Unknown"
                        
            except Exception as e:
                print(f"ERROR loading data: {e}")
                self._df = pd.DataFrame()
        
        return self._df
    
    def _safe_filter(self, df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
        """
        Safely filter DataFrame by column value with case-insensitive matching
        """
        if df.empty or column not in df.columns:
            return df
        
        if value is None or value == "":
            return df
        
        try:
            # Case-insensitive string matching for objects
            if df[column].dtype == 'object':
                return df[df[column].astype(str).str.contains(
                    str(value), na=False, case=False
                )]
            else:
                return df[df[column] == value]
        except Exception as e:
            print(f"Warning: Error filtering {column} with value {value}: {e}")
            return df
    
    def get_bookings(
        self, 
        status: Optional[str] = None,
        agent_name: Optional[str] = None,
        pax_name: Optional[str] = None,
        ref_no: Optional[str] = None,
        city: Optional[str] = None,
        cancellation_deadline: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Retrieve bookings with filters mapping to the Excel columns
        """
        try:
            df = self._load_data()
            
            if df.empty:
                return {
                    "success": False,
                    "error": "No booking data available",
                    "data": [],
                    "count": 0
                }
            
            # Apply filters based on provided column names
            if status:
                # Check both the code and the label
                df_code = self._safe_filter(df, 'Status', status)
                df_label = self._safe_filter(df, 'Status_Label', status)
                df = pd.concat([df_code, df_label]).drop_duplicates()
            
            if agent_name:
                # Map 'agent_name' to 'Client Name' or 'Created By' or 'Branch' as appropriate
                df = self._safe_filter(df, 'Client Name', agent_name)
            
            if pax_name:
                df = self._safe_filter(df, 'Lead Pax Name', pax_name)
                
            if ref_no:
                df = self._safe_filter(df, 'Reference No', ref_no)
                
            if city:
                df = self._safe_filter(df, 'City', city)
            
            if cancellation_deadline:
                df = self._safe_filter(df, 'Cancellation Deadline', cancellation_deadline)
            
            # Limit results
            df_display = df.head(limit)
            
            # Convert to list of dicts
            records = df_display.to_dict('records')
            
            # Clean up records for JSON serialization (handle NaT/NaN)
            records = self._clean_records(records)
            
            return {
                "success": True,
                "data": records,
                "count": len(df),
                "display_count": len(records),
                "message": f"Found {len(df)} booking(s)"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error retrieving bookings: {str(e)}",
                "data": [],
                "count": 0
            }

    def _clean_records(self, records: List[Dict]) -> List[Dict]:
        """Handle NaN and NaT for JSON serialization"""
        import numpy as np
        cleaned = []
        for rec in records:
            new_rec = {}
            for k, v in rec.items():
                if pd.isna(v):
                    new_rec[k] = None
                elif isinstance(v, (datetime, pd.Timestamp)):
                    new_rec[k] = v.strftime('%Y-%m-%d')
                else:
                    new_rec[k] = v
            cleaned.append(new_rec)
        return cleaned

    def get_column_names(self) -> List[str]:
        """Get list of column names from the data"""
        df = self._load_data()
        return df.columns.tolist()

# Singleton instance
_data_service_instance = None

def get_data_service() -> DataService:
    """Get or create singleton data service instance"""
    global _data_service_instance
    if _data_service_instance is None:
        _data_service_instance = DataService()
    return _data_service_instance
