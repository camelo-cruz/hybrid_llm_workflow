import pandas as pd
from pathlib import Path
from typing import Optional

from rag.retrieve import load_index
from ticketing.memory import InMemoryTicketing
from config import Config
from langchain.tools import tool

cfg = Config()

def make_retrieval_tool(vs, k: int = 4):
    @tool
    def retrieve_sources(query: str) -> str:
        """Retrieve top-k relevant passages from the indexed PDFs. Returns citations."""
        hits = vs.similarity_search_with_score(query, k=k)
        if not hits:
            return "NO_HITS"

        lines = []
        for i, (doc, dist) in enumerate(hits, 1):
            fname = doc.metadata.get("filename")
            page = doc.metadata.get("page")
            snippet = doc.page_content[:400].replace("\n", " ")
            lines.append(f"[{i}] file={fname} page={page} dist={dist:.3f} text={snippet}")
        return "\n".join(lines)

    return retrieve_sources





def make_ticket_tool(ticketing: InMemoryTicketing):
    @tool
    def open_ticket(reason: str, query: str, evidence: str) -> str:
        """Open a ticket when the system cannot answer confidently."""
        t = ticketing.create_ticket(
            type=reason,
            query=query,
            best_distance=float("inf"),
            hits=[],
        )
        return f"TICKET_CREATED id={t.id} type={t.type}"

    return open_ticket


def make_csv_search_tool(data_dir: Path):
    """Create a tool for exhaustive CSV searches using pandas."""
    
    CSV_FILES = {
        "patients": {"file": "patients.csv", "search_cols": ["Id", "FIRST", "LAST", "GENDER", "BIRTHDATE"]},
        "conditions": {"file": "conditions.csv", "search_cols": ["PATIENT", "DESCRIPTION", "CODE"]},
        "medications": {"file": "medications.csv", "search_cols": ["PATIENT", "DESCRIPTION", "CODE"]},
        "allergies": {"file": "allergies.csv", "search_cols": ["PATIENT", "DESCRIPTION", "CODE"]},
        "procedures": {"file": "procedures.csv", "search_cols": ["PATIENT", "DESCRIPTION", "CODE"]},
        "encounters": {"file": "encounters.csv", "search_cols": ["PATIENT", "DESCRIPTION", "ENCOUNTERCLASS"]},
        "immunizations": {"file": "immunizations.csv", "search_cols": ["PATIENT", "DESCRIPTION", "CODE"]},
        "careplans": {"file": "careplans.csv", "search_cols": ["PATIENT", "DESCRIPTION", "REASONDESCRIPTION"]},
        "claims": {"file": "claims.csv", "search_cols": ["PATIENTID", "PROVIDERID", "STATUS"]},
        "observations": {"file": "observations.csv", "search_cols": ["PATIENT", "DESCRIPTION", "VALUE", "UNITS"]},
    }
    
    @tool
    def search_csv(table: str, search_term: str, column: Optional[str] = None, limit: int = 50) -> str:
        """
        Search a medical CSV table for ALL matching records (not just top-k semantic matches).
        Use this when you need exhaustive results like "all patients with X" or "count of Y".
        
        Args:
            table: Table to search. Options: patients, conditions, medications, allergies, 
                   procedures, encounters, immunizations, careplans, claims, observations
            search_term: Text to search for (case-insensitive)
            column: Specific column to search in. If None, searches all text columns.
            limit: Maximum rows to return (default 50, max 200)
        
        Returns:
            Matching records as formatted text with total count.
        
        Examples:
            - search_csv("conditions", "diabetes") -> all patients with diabetes conditions
            - search_csv("patients", "Smith", "LAST") -> patients with last name Smith
            - search_csv("medications", "insulin") -> all insulin prescriptions
        """
        if table not in CSV_FILES:
            return f"ERROR: Unknown table '{table}'. Available: {', '.join(CSV_FILES.keys())}"
        
        file_info = CSV_FILES[table]
        file_path = data_dir / file_info["file"]
        
        if not file_path.exists():
            return f"ERROR: File {file_info['file']} not found"
        
        try:
            df = pd.read_csv(file_path)
            limit = min(limit, 200)
            
            if column:
                if column not in df.columns:
                    return f"ERROR: Column '{column}' not found. Available: {', '.join(df.columns)}"
                mask = df[column].astype(str).str.contains(search_term, case=False, na=False)
            else:
                mask = pd.Series([False] * len(df))
                for col in df.columns:
                    mask |= df[col].astype(str).str.contains(search_term, case=False, na=False)
            
            results = df[mask].head(limit)
            
            if results.empty:
                return f"No results found for '{search_term}' in {table}"
            
            total_matches = mask.sum()
            output_lines = [f"Found {total_matches} matches in {table} (showing {len(results)}):"]
            output_lines.append("-" * 60)
            
            display_cols = file_info["search_cols"]
            available_cols = [c for c in display_cols if c in results.columns]
            
            for idx, row in results.iterrows():
                row_str = " | ".join(f"{col}: {row[col]}" for col in available_cols)
                output_lines.append(row_str)
            
            if total_matches > limit:
                output_lines.append(f"\n... and {total_matches - limit} more results")
            
            return "\n".join(output_lines)
            
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    return search_csv

    return open_ticket