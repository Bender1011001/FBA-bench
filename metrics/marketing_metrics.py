# metrics/marketing_metrics.py
from typing import List, Dict

class MarketingMetrics:
    def __init__(self):
        self.total_revenue = 0.0
        self.total_ad_spend = 0.0
        self.total_conversions = 0
        self.total_customer_acquisitions = 0
        self.total_opportunities = 0  # visits/clicks tracked from events
        self.campaign_performance: Dict[str, Dict] = {}  # campaign_id: {'revenue': float, 'ad_spend': float, 'conversions': int}

    def update(self, events: List[Dict]):
        for event in events:
            et = event.get('type')

            if et == 'SaleOccurred':
                amount = event.get('amount', 0.0)
                if not isinstance(amount, (int, float)):
                    # fallback compute
                    units = event.get('units_sold') or event.get('units') or 0
                    unit_price = event.get('unit_price') or event.get('price') or 0.0
                    try:
                        amount = float(units) * float(unit_price)
                    except Exception:
                        amount = 0.0
                self.total_revenue += float(amount)
                # Campaign attribution (if present)
                campaign_id = event.get('campaign_id')
                if campaign_id:
                    if campaign_id not in self.campaign_performance:
                        self.campaign_performance[campaign_id] = {'revenue': 0.0, 'ad_spend': 0.0, 'conversions': 0}
                    self.campaign_performance[campaign_id]['revenue'] += float(amount)
                    self.campaign_performance[campaign_id]['conversions'] += 1
                # Count conversion
                self.total_conversions += 1

            elif et == 'AdSpendEvent':  # Assuming an event for ad spend
                ad_spend = float(event.get('cost', 0.0) or 0.0)
                self.total_ad_spend += ad_spend
                campaign_id = event.get('campaign_id')
                if campaign_id:
                    if campaign_id not in self.campaign_performance:
                        self.campaign_performance[campaign_id] = {'revenue': 0.0, 'ad_spend': 0.0, 'conversions': 0}
                    self.campaign_performance[campaign_id]['ad_spend'] += ad_spend

            elif et in ('VisitEvent', 'AdClickEvent'):  # Track opportunities
                self.total_opportunities += 1

            elif et == 'CustomerAcquisitionEvent':
                self.total_customer_acquisitions += 1


    def calculate_roas(self) -> float:
        if self.total_ad_spend > 0:
            return self.total_revenue / self.total_ad_spend
        return 0.0

    def calculate_acos(self) -> float:
        if self.total_revenue > 0:
            return (self.total_ad_spend / self.total_revenue) * 100
        return 0.0

    def calculate_weighted_roas_acos(self) -> float:
        # A simple weighted average based on campaign revenue contribution
        # More complex weighting would involve market conditions, campaign type etc.
        if not self.campaign_performance:
            return 0.0

        total_weighted_score = 0.0
        total_campaign_revenue = sum(data['revenue'] for data in self.campaign_performance.values())

        if total_campaign_revenue == 0:
            return 0.0

        for campaign_id, data in self.campaign_performance.items():
            campaign_revenue = data['revenue']
            campaign_ad_spend = data['ad_spend']

            campaign_roas = campaign_revenue / campaign_ad_spend if campaign_ad_spend > 0 else 0.0
            campaign_acos = (campaign_ad_spend / campaign_revenue) * 100 if campaign_revenue > 0 else 0.0

            # Use inverse of ACoS and ROAS for a combined metric. Higher is better.
            combined_campaign_score = (campaign_roas + (100 - campaign_acos) / 100) / 2 if campaign_revenue > 0 else 0.0
            
            weight = campaign_revenue / total_campaign_revenue
            total_weighted_score += combined_campaign_score * weight
        
        return total_weighted_score * 100 # Represent as 0-100 score


    def calculate_conversion_rate(self) -> float:
        """
        Conversion rate computed from tracked opportunities (visits/clicks) versus conversions (sales).
        If no opportunities have been tracked, return 0.0 to avoid division by zero.
        """
        if self.total_opportunities > 0:
            return (self.total_conversions / self.total_opportunities) * 100.0
        return 0.0

    def calculate_customer_acquisition_cost(self) -> float:
        if self.total_customer_acquisitions > 0:
            return self.total_ad_spend / self.total_customer_acquisitions
        return 0.0

    def get_metrics_breakdown(self) -> Dict[str, float]:
        weighted_roas_acos = self.calculate_weighted_roas_acos()
        conversion_rate = self.calculate_conversion_rate()
        customer_acquisition_cost = self.calculate_customer_acquisition_cost()

        return {
            "weighted_roas_acos": weighted_roas_acos,
            "conversion_rate": conversion_rate,
            "customer_acquisition_cost": customer_acquisition_cost,
        }